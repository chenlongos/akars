use crate::linux;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

const CVI_CAMERA_IOCTL_INIT: u64 = 1;
const CVI_CAMERA_IOCTL_GET_INFO: u64 = 2;
// cmd 4: kernel returns a raw YUV422 planar (I422) frame instead of MJPEG.
const CVI_CAMERA_IOCTL_GET_FRAME: u64 = 4;
// I422 (YUV422 planar) frame layout: full-size Y plane + U and V planes each
// subsampled horizontally (w/2 × h), i.e. `width * height * 2` bytes.
// The concrete size is derived in `get_frame` from the resolution the kernel
// actually negotiated (reported by GET_INFO) instead of a hardcoded 640×480, so
// cameras without a 640×480 mode (e.g. b349:b182 → 640×360) also work while a
// 640×480 camera is unaffected.
const YUV_BYTES_PER_PIXEL: usize = 2;
const JPEG_MARKER_START: [u8; 2] = [0xFF, 0xD8];
const JPEG_MARKER_END: [u8; 2] = [0xFF, 0xD9];

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct RawCameraInfo {
    width: u16,
    height: u16,
    format: u8,
    connected: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraInfo {
    pub width: u16,
    pub height: u16,
    pub format: u8,
    pub connected: bool,
}

impl RawCameraInfo {
    fn unpack(&self) -> CameraInfo {
        let width = unsafe { std::ptr::addr_of!(self.width).read_unaligned() };
        let height = unsafe { std::ptr::addr_of!(self.height).read_unaligned() };
        CameraInfo {
            width,
            height,
            format: self.format,
            connected: self.connected != 0,
        }
    }
}

/// 帧的平面格式。本分支的取帧路径固定返回 YUV422 planar,即 `I422`;
/// `I420` 仅用于让 `bucket` 的策略代码能表达两种布局。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarFormat {
    /// YUV420 planar: 640×480×1.5 = 460800 字节。
    I420,
    /// YUV422 planar: 640×480×2 = 614400 字节。
    I422,
}

#[derive(Debug)]
pub struct CameraFrame {
    /// Raw YUV422 planar (I422) pixels: Y plane, then U plane, then V plane.
    pub pixels: Vec<u8>,
    pub width: u16,
    pub height: u16,
}

pub struct UsbCamera {
    file: File,
    info: CameraInfo,
}

impl UsbCamera {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(linux::O_NOCTTY)
            .open(path.as_ref())?;

        ioctl_no_arg(file.as_raw_fd(), CVI_CAMERA_IOCTL_INIT)?;

        let mut raw = RawCameraInfo::default();
        ioctl_ptr(
            file.as_raw_fd(),
            CVI_CAMERA_IOCTL_GET_INFO,
            &mut raw as *mut RawCameraInfo,
        )?;
        let info = raw.unpack();
        if !info.connected {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "camera reports disconnected",
            ));
        }
        if info.format != 1 {
            eprintln!(
                "[camera] warning: camera format {} is not MJPEG(1)",
                info.format
            );
        }

        Ok(Self { file, info })
    }

    pub fn info(&self) -> CameraInfo {
        self.info
    }

    pub fn get_frame(&mut self) -> io::Result<CameraFrame> {
        // Size the buffer from the negotiated resolution reported at open time.
        // The kernel decodes MJPEG to I422 (`width * height * 2` bytes) at that
        // resolution, so a 640×480 camera keeps the old 614400-byte path while a
        // 640×360 camera (no 640×480 mode) gets a correctly-sized 460800-byte frame.
        let width = self.info.width as usize;
        let height = self.info.height as usize;
        let expected = width * height * YUV_BYTES_PER_PIXEL;
        if expected == 0 {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "camera reported a zero-sized frame",
            ));
        }

        let mut buffer = vec![0u8; expected];
        let ret = unsafe {
            linux::ioctl(
                self.file.as_raw_fd(),
                CVI_CAMERA_IOCTL_GET_FRAME as _,
                buffer.as_mut_ptr(),
            )
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }

        // eprintln!("[camera] ioctl returned {ret} bytes");
        // The kernel may signal success with 0 or report the byte count; either
        // way we expect one full I422 frame in the buffer.
        let size = ret as usize;

        if size != 0 && size < expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "camera returned a short YUV frame",
            ));
        }
        buffer.truncate(expected);

        Ok(CameraFrame {
            pixels: buffer,
            width: self.info.width,
            height: self.info.height,
        })
    }
}

fn ioctl_no_arg(fd: i32, request: u64) -> io::Result<()> {
    let ret = unsafe { linux::ioctl(fd, request as _, 0usize) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn ioctl_ptr<T>(fd: i32, request: u64, ptr: *mut T) -> io::Result<()> {
    let ret = unsafe { linux::ioctl(fd, request as _, ptr) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn is_valid_jpeg(data: &[u8]) -> bool {
    data.len() >= 4 && data.starts_with(&JPEG_MARKER_START) && data.ends_with(&JPEG_MARKER_END)
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
