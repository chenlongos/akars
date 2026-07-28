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
const YUV_FRAME_WIDTH: usize = 640;
const YUV_FRAME_HEIGHT: usize = 480;
// I422: full-size Y plane + U and V planes each subsampled horizontally (w/2 × h).
const YUV_FRAME_SIZE: usize = YUV_FRAME_WIDTH * YUV_FRAME_HEIGHT * 2;
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
        let mut buffer = vec![0u8; YUV_FRAME_SIZE];
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
        // way we expect one full I420 frame in the buffer.
        let size = ret as usize;

        if size != 0 && size < YUV_FRAME_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "camera returned a short YUV frame",
            ));
        }
        buffer.truncate(YUV_FRAME_SIZE);

        Ok(CameraFrame {
            pixels: buffer,
            width: YUV_FRAME_WIDTH as u16,
            height: YUV_FRAME_HEIGHT as u16,
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
