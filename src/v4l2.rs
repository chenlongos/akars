//! 标准 Linux V4L2 接口的 Rust 封装。
//!
//! 提供 V4L2 所需的数据结构、常量和 ioctl/mmap/poll 封装。
//! 数据结构严格按 <linux/videodev2.h> 定义，大小已用 Rust/C 交叉验证。

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;
use std::io;
use std::os::raw::{c_int, c_long, c_ulong};

// ─── open() flags ────────────────────────────────────────────────────────────

pub const O_NONBLOCK: i32 = 0o4000;

// ─── V4L2 常量 ───────────────────────────────────────────────────────────────

pub const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
pub const V4L2_MEMORY_MMAP: u32 = 1;
pub const V4L2_FIELD_ANY: u32 = 0;
pub const V4L2_COLORSPACE_SRGB: u32 = 8;

pub const V4L2_PIX_FMT_MJPEG: u32 = 0x47504a4d; // v4l2_fourcc('M','J','P','G')
pub const V4L2_PIX_FMT_YUYV: u32 = 0x56595559; // v4l2_fourcc('Y','U','Y','V')

pub const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x1;
pub const V4L2_CAP_STREAMING: u32 = 0x4000000;
pub const V4L2_CAP_TIMEPERFRAME: u32 = 0x1000;

// ─── Ioctl 命令码（x86_64 Linux，gcc 计算得到） ─────────────────────────────

pub const VIDIOC_QUERYCAP: c_ulong = 0x80685600;
pub const VIDIOC_G_FMT: c_ulong = 0xc0d05604;
pub const VIDIOC_S_FMT: c_ulong = 0xc0d05605;
pub const VIDIOC_REQBUFS: c_ulong = 0xc0145608;
pub const VIDIOC_QUERYBUF: c_ulong = 0xc0585609;
pub const VIDIOC_QBUF: c_ulong = 0xc058560f;
pub const VIDIOC_DQBUF: c_ulong = 0xc0585611;
pub const VIDIOC_STREAMON: c_ulong = 0x40045612;
pub const VIDIOC_STREAMOFF: c_ulong = 0x40045613;
pub const VIDIOC_G_PARM: c_ulong = 0xc0cc5615;
pub const VIDIOC_S_PARM: c_ulong = 0xc0cc5616;

// ─── 数据结构 ────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct v4l2_capability {
    pub driver: [u8; 16],
    pub card: [u8; 32],
    pub bus_info: [u8; 32],
    pub version: u32,
    pub capabilities: u32,
    pub device_caps: u32,
    pub reserved: [u32; 3],
}
// size check: 104 ✓

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct v4l2_pix_format {
    pub width: u32,
    pub height: u32,
    pub pixelformat: u32,
    pub field: u32,
    pub bytesperline: u32,
    pub sizeimage: u32,
    pub colorspace: u32,
    pub priv_: u32,
    pub flags: u32,
    pub ycbcr_enc: u32,
    pub quantization: u32,
    pub xfer_func: u32,
}
// size check: 48 ✓

/// v4l2_format 的 fmt union 包含指针类型成员 (v4l2_window.rects)，
/// 因此在 x86_64 上对齐要求是 8 字节。显式标注 align(8)，
/// 避免 Rust 仅根据我们声明的成员（pix + raw_data）推断对齐为 4。
#[repr(C, align(8))]
pub union v4l2_format_fmt {
    pub pix: v4l2_pix_format,
    pub raw_data: [u8; 200],
}
// union alignment = 8 ✓（加上显式标注）

#[repr(C)]
pub struct v4l2_format {
    pub type_: u32,
    /// fmt 的对齐是 8，所以 type_ 之后需要 4 字节 padding 才能放在 offset 8。
    pub fmt: v4l2_format_fmt,
}
// size check: 208 ✓  (type_ at 0, padding 4 bytes, fmt at 8, size 200 → 208)

// 手动实现 Default，因为 union 不能 derive
unsafe impl Sync for v4l2_format_fmt {}
unsafe impl Send for v4l2_format_fmt {}
unsafe impl Sync for v4l2_format {}
unsafe impl Send for v4l2_format {}

impl Default for v4l2_format_fmt {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
impl Default for v4l2_format {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct v4l2_requestbuffers {
    pub count: u32,
    pub type_: u32,
    pub memory: u32,
    pub capabilities: u32,
    pub flags: u8,
    pub reserved: [u8; 3],
}
// size check: 20 ✓

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct v4l2_timecode {
    pub type_: u32,
    pub flags: u32,
    pub frames: u8,
    pub seconds: u8,
    pub minutes: u8,
    pub hours: u8,
    pub userbits: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct timeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

#[repr(C)]
pub union v4l2_buffer_m {
    pub offset: u32,
    pub userptr: u64,
    pub planes: *mut c_void,
    pub fd: i32,
}
unsafe impl Sync for v4l2_buffer_m {}
unsafe impl Send for v4l2_buffer_m {}

// 手动 Clone/Copy，union 不能 derive
impl Clone for v4l2_buffer_m {
    fn clone(&self) -> Self { *self }
}
impl Copy for v4l2_buffer_m {}
impl Default for v4l2_buffer_m {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct v4l2_buffer {
    pub index: u32,
    pub type_: u32,
    pub bytesused: u32,
    pub flags: u32,
    pub field: u32,
    pub timestamp: timeval,
    pub timecode: v4l2_timecode,
    pub sequence: u32,
    pub memory: u32,
    pub m: v4l2_buffer_m,
    pub length: u32,
    pub reserved2: u32,
    pub request_fd: i32,
}
unsafe impl Sync for v4l2_buffer {}
unsafe impl Send for v4l2_buffer {}
impl Default for v4l2_buffer {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
// size check: 88 ✓

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct v4l2_fract {
    pub numerator: u32,
    pub denominator: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct v4l2_captureparm {
    pub capability: u32,
    pub capturemode: u32,
    pub timeperframe: v4l2_fract,
    pub extendedmode: u32,
    pub readbuffers: u32,
    pub reserved: [u32; 4],
}

#[repr(C)]
pub union v4l2_streamparm_parm {
    pub capture: v4l2_captureparm,
    pub raw_data: [u8; 200],
}

#[repr(C)]
pub struct v4l2_streamparm {
    pub type_: u32,
    pub parm: v4l2_streamparm_parm,
}
// size check: 204 ✓  (type_ at 0, parm at 4, union align=4 → total 4+200=204)

unsafe impl Sync for v4l2_streamparm_parm {}
unsafe impl Send for v4l2_streamparm_parm {}
unsafe impl Sync for v4l2_streamparm {}
unsafe impl Send for v4l2_streamparm {}
impl Default for v4l2_streamparm_parm {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}
impl Default for v4l2_streamparm {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

// ─── Ioctl 封装 ──────────────────────────────────────────────────────────────

/// 执行 ioctl 系统调用，arg 是指向参数的裸指针。
pub fn ioctl_ptr<T>(fd: i32, request: c_ulong, ptr: *mut T) -> io::Result<i32> {
    let ret = unsafe { raw_ioctl(fd, request, ptr as *mut c_void) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret)
    }
}

/// ioctl 不带参数（实际传 0）。
pub fn ioctl_noarg(fd: i32, request: c_ulong) -> io::Result<i32> {
    let ret = unsafe { raw_ioctl(fd, request, std::ptr::null_mut::<c_void>()) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret)
    }
}

/// VIDIOC_STREAMON / STREAMOFF 需要一个指向 u32 的指针。
pub fn ioctl_u32_ptr(fd: i32, request: c_ulong, value: u32) -> io::Result<i32> {
    let val = value;
    ioctl_ptr(fd, request, &val as *const u32 as *mut u32)
}

unsafe extern "C" {
    #[link_name = "ioctl"]
    fn raw_ioctl(fd: c_int, request: c_ulong, arg: *mut c_void) -> c_int;
}

// ─── mmap 封装 ───────────────────────────────────────────────────────────────

const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;
const MAP_FAILED: *mut c_void = -1_isize as *mut c_void;

unsafe extern "C" {
    #[link_name = "mmap"]
    fn raw_mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: c_long,
    ) -> *mut c_void;

    #[link_name = "munmap"]
    fn raw_munmap(addr: *mut c_void, length: usize) -> c_int;
}

pub fn mmap_buffer(fd: i32, length: usize, offset: u64) -> io::Result<*mut c_void> {
    let ret = unsafe {
        raw_mmap(
            std::ptr::null_mut(),
            length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            offset as c_long,
        )
    };
    if ret == MAP_FAILED {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret)
    }
}

pub fn munmap_buffer(addr: *mut c_void, length: usize) -> io::Result<()> {
    let ret = unsafe { raw_munmap(addr, length) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

// ─── poll 封装 ───────────────────────────────────────────────────────────────

pub const POLLIN: i16 = 0x001;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pollfd {
    pub fd: i32,
    pub events: i16,
    pub revents: i16,
}

unsafe extern "C" {
    #[link_name = "poll"]
    fn raw_poll(fds: *mut pollfd, nfds: u64, timeout: i32) -> c_int;
}

pub fn poll(fds: &mut [pollfd], timeout_ms: i32) -> io::Result<i32> {
    let ret = unsafe { raw_poll(fds.as_mut_ptr(), fds.len() as u64, timeout_ms) };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret)
    }
}
