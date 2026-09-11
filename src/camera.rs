//! V4L2 摄像头采集（mmap + poll 模式）。
//!
//! 与 C++ 示例（`example_of_softlink/csrc/src/camera.cpp`）采用完全相同的
//! V4L2 流程：QUERYCAP → 协商格式 (YUYV 优先, MJPEG 回退) → REQBUFS →
//! QUERYBUF/mmap → QBUF 预热 → STREAMON → poll + DQBUF/QBUF 取流。

use crate::v4l2;
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

const V4L2_NUM_BUFFERS: u32 = 3;
const POLL_TIMEOUT_MS: i32 = 1000;

#[derive(Clone, Copy, Debug)]
pub struct CameraInfo {
    pub width: u32,
    pub height: u32,
    pub pixelformat: u32,
}

impl CameraInfo {
    pub fn format_name(&self) -> &'static str {
        match self.pixelformat {
            v4l2::V4L2_PIX_FMT_MJPEG => "MJPEG",
            v4l2::V4L2_PIX_FMT_YUYV => "YUYV",
            _ => "unknown",
        }
    }
}

#[derive(Debug)]
pub struct CameraFrame {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

struct MmapBuf {
    ptr: *mut c_void,
    len: usize,
}
unsafe impl Send for MmapBuf {}
unsafe impl Sync for MmapBuf {}

pub struct UsbCamera {
    file: File,
    info: CameraInfo,
    buffers: Vec<MmapBuf>,
    running: bool,
}

impl UsbCamera {
    pub fn open<P: AsRef<Path>>(path: P, mut width: u32, mut height: u32) -> io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(v4l2::O_NONBLOCK)
            .open(path.as_ref())?;
        let fd = file.as_raw_fd();

        // ── 1. QUERYCAP ──────────────────────────────────────────────────
        let mut cap = v4l2::v4l2_capability::default();
        v4l2::ioctl_ptr(fd, v4l2::VIDIOC_QUERYCAP, &mut cap)?;
        if (cap.capabilities & v4l2::V4L2_CAP_VIDEO_CAPTURE) == 0 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "device is not a video capture device",
            ));
        }
        if (cap.capabilities & v4l2::V4L2_CAP_STREAMING) == 0 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "device does not support streaming",
            ));
        }
        let driver = {
            let nul = cap.driver.iter().position(|&b| b == 0).unwrap_or(cap.driver.len());
            String::from_utf8_lossy(&cap.driver[..nul]).into_owned()
        };
        let card = {
            let nul = cap.card.iter().position(|&b| b == 0).unwrap_or(cap.card.len());
            String::from_utf8_lossy(&cap.card[..nul]).into_owned()
        };
        eprintln!("[camera] V4L2 driver={driver} card={card}");

        // ── 2. 协商格式（YUYV 优先, MJPEG 回退） ────────────────────────
        // YUYV→RGB 是纯 CPU 计算（比 MJPEG 解码快得多），先试 YUYV。
        let requested_w = width;
        let requested_h = height;
        let mut pixelformat = v4l2::V4L2_PIX_FMT_YUYV;

        let mut fmt_set = try_set_format(fd, pixelformat, requested_w, requested_h)?;
        if fmt_set.pixelformat != pixelformat {
            eprintln!("[camera] YUYV not supported, trying MJPEG");
            pixelformat = v4l2::V4L2_PIX_FMT_MJPEG;
            fmt_set = try_set_format(fd, pixelformat, requested_w, requested_h)?;
        }
        width = fmt_set.width;
        height = fmt_set.height;
        pixelformat = fmt_set.pixelformat;
        eprintln!(
            "[camera] negotiated {} {}x{} sizeimage={}",
            fmt_name(pixelformat),
            width,
            height,
            fmt_set.sizeimage
        );

        // ── 3. REQBUFS ──────────────────────────────────────────────────
        let mut req = v4l2::v4l2_requestbuffers::default();
        req.count = V4L2_NUM_BUFFERS;
        req.type_ = v4l2::V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = v4l2::V4L2_MEMORY_MMAP;
        v4l2::ioctl_ptr(fd, v4l2::VIDIOC_REQBUFS, &mut req)?;
        if req.count < 2 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("REQBUFS returned only {} buffers", req.count),
            ));
        }
        eprintln!("[camera] REQBUFS count={}", req.count);

        // ── 4. QUERYBUF + mmap ───────────────────────────────────────────
        let mut buffers = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let mut buf = v4l2::v4l2_buffer::default();
            buf.index = i;
            buf.type_ = v4l2::V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = v4l2::V4L2_MEMORY_MMAP;
            v4l2::ioctl_ptr(fd, v4l2::VIDIOC_QUERYBUF, &mut buf)?;
            let offset = unsafe { buf.m.offset } as u64;
            let ptr = v4l2::mmap_buffer(fd, buf.length as usize, offset)?;
            buffers.push(MmapBuf {
                ptr,
                len: buf.length as usize,
            });
        }

        // ── 5. QBUF 预热 ────────────────────────────────────────────────
        for i in 0..req.count {
            let mut buf = v4l2::v4l2_buffer::default();
            buf.index = i;
            buf.type_ = v4l2::V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = v4l2::V4L2_MEMORY_MMAP;
            v4l2::ioctl_ptr(fd, v4l2::VIDIOC_QBUF, &mut buf)?;
        }

        // ── 6. STREAMON ─────────────────────────────────────────────────
        v4l2::ioctl_u32_ptr(
            fd,
            v4l2::VIDIOC_STREAMON,
            v4l2::V4L2_BUF_TYPE_VIDEO_CAPTURE,
        )?;
        eprintln!("[camera] stream ON");

        Ok(Self {
            file,
            info: CameraInfo {
                width,
                height,
                pixelformat,
            },
            buffers,
            running: true,
        })
    }

    pub fn info(&self) -> CameraInfo {
        self.info
    }

    pub fn get_frame(&mut self) -> io::Result<CameraFrame> {
        let fd = self.file.as_raw_fd();
        loop {
            let pollfd = v4l2::pollfd {
                fd,
                events: v4l2::POLLIN,
                revents: 0,
            };
            let ret = v4l2::poll(&mut [pollfd], POLL_TIMEOUT_MS)?;
            if ret == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "camera poll timeout",
                ));
            }

            let mut buf = v4l2::v4l2_buffer::default();
            buf.type_ = v4l2::V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = v4l2::V4L2_MEMORY_MMAP;
            let dq_ret = v4l2::ioctl_ptr(fd, v4l2::VIDIOC_DQBUF, &mut buf);
            if let Err(ref e) = dq_ret {
                if e.kind() == io::ErrorKind::WouldBlock {
                    continue;
                }
                dq_ret?;
            }

            let idx = buf.index as usize;
            if idx >= self.buffers.len() {
                let _ = v4l2::ioctl_ptr(fd, v4l2::VIDIOC_QBUF, &mut buf);
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("DQBUF index {} out of range (nbufs={})", idx, self.buffers.len()),
                ));
            }

            let mmap_buf = &self.buffers[idx];
            let bytesused = buf.bytesused as usize;
            if bytesused == 0 || bytesused > mmap_buf.len {
                let _ = v4l2::ioctl_ptr(fd, v4l2::VIDIOC_QBUF, &mut buf);
                continue;
            }

            let data_ptr = mmap_buf.ptr as *const u8;
            let mut pixels = vec![0u8; bytesused];
            unsafe {
                std::ptr::copy_nonoverlapping(data_ptr, pixels.as_mut_ptr(), bytesused);
            }

            let _ = v4l2::ioctl_ptr(fd, v4l2::VIDIOC_QBUF, &mut buf);

            return Ok(CameraFrame {
                pixels,
                width: self.info.width,
                height: self.info.height,
            });
        }
    }
}

impl Drop for UsbCamera {
    fn drop(&mut self) {
        if self.running {
            let fd = self.file.as_raw_fd();
            let _ = v4l2::ioctl_u32_ptr(
                fd,
                v4l2::VIDIOC_STREAMOFF,
                v4l2::V4L2_BUF_TYPE_VIDEO_CAPTURE,
            );
            self.running = false;
        }
        for buf in self.buffers.drain(..) {
            let _ = v4l2::munmap_buffer(buf.ptr, buf.len);
        }
    }
}

// ── 辅助函数 ─────────────────────────────────────────────────────────────────

fn try_set_format(
    fd: i32,
    pixelformat: u32,
    width: u32,
    height: u32,
) -> io::Result<v4l2::v4l2_pix_format> {
    let mut fmt = v4l2::v4l2_format::default();
    fmt.type_ = v4l2::V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = v4l2::V4L2_FIELD_ANY;
    fmt.fmt.pix.colorspace = v4l2::V4L2_COLORSPACE_SRGB;

    v4l2::ioctl_ptr(fd, v4l2::VIDIOC_S_FMT, &mut fmt)?;
    Ok(unsafe { fmt.fmt.pix })
}

fn fmt_name(pixelformat: u32) -> &'static str {
    match pixelformat {
        v4l2::V4L2_PIX_FMT_MJPEG => "MJPEG",
        v4l2::V4L2_PIX_FMT_YUYV => "YUYV",
        _ => "unknown",
    }
}

pub fn is_valid_jpeg(data: &[u8]) -> bool {
    data.len() >= 4
        && data.starts_with(&[0xFF, 0xD8])
        && data.ends_with(&[0xFF, 0xD9])
}

#[cfg(test)]
mod tests {
    use super::is_valid_jpeg;

    #[test]
    fn checks_jpeg_markers() {
        assert!(is_valid_jpeg(&[0xFF, 0xD8, 1, 2, 0xFF, 0xD9]));
        assert!(!is_valid_jpeg(&[0x00, 0xD8, 1, 2, 0xFF, 0xD9]));
        assert!(!is_valid_jpeg(&[0xFF, 0xD8, 1]));
    }
}
