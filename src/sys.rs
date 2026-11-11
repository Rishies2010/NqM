use crate::buf::AlignedBuf;
use crate::error::{Error, Result};
use alloc::alloc::{GlobalAlloc, Layout};
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use core::fmt;

#[link(name = "c")]
extern "C" {
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, n: usize) -> isize;
    fn write(fd: c_int, buf: *const c_void, n: usize) -> isize;
    fn lseek(fd: c_int, off: isize, whence: c_int) -> isize;
    fn close(fd: c_int) -> c_int;
    fn malloc(n: usize) -> *mut c_void;
    fn free(p: *mut c_void);
    fn realloc(p: *mut c_void, n: usize) -> *mut c_void;
    pub fn abort() -> !;
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct Timespec {
    sec: i64,
    nsec: i64,
}
#[cfg(target_os = "linux")]
extern "C" {
    fn clock_gettime(clk: c_int, ts: *mut Timespec) -> c_int;
}

pub fn now_ms() -> u64 {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut ts = Timespec { sec: 0, nsec: 0 };
        if clock_gettime(1, &mut ts) == 0 {
            return ts.sec as u64 * 1000 + ts.nsec as u64 / 1_000_000;
        }
    }
    0
}

const MIN_ALIGN: usize = 8;

pub struct LibcAlloc;

unsafe impl GlobalAlloc for LibcAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.align() <= MIN_ALIGN {
            return malloc(l.size().max(1)) as *mut u8;
        }
        let word = core::mem::size_of::<usize>();
        let raw = malloc(l.size() + l.align() + word) as usize;
        if raw == 0 {
            return core::ptr::null_mut();
        }
        let aligned = (raw + word + l.align() - 1) & !(l.align() - 1);
        *((aligned - word) as *mut usize) = raw;
        aligned as *mut u8
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if l.align() <= MIN_ALIGN {
            free(p as *mut c_void);
        } else {
            let raw = *((p as usize - core::mem::size_of::<usize>()) as *const usize);
            free(raw as *mut c_void);
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        if l.align() <= MIN_ALIGN {
            return realloc(p as *mut c_void, new_size.max(1)) as *mut u8;
        }
        let nl = Layout::from_size_align_unchecked(new_size, l.align());
        let np = self.alloc(nl);
        if !np.is_null() {
            core::ptr::copy_nonoverlapping(p, np, l.size().min(new_size));
            self.dealloc(p, l);
        }
        np
    }
}

pub fn write_all(fd: i32, mut b: &[u8]) {
    while !b.is_empty() {
        let n = unsafe { write(fd, b.as_ptr() as *const c_void, b.len()) };
        if n <= 0 {
            return;
        }
        b = &b[n as usize..];
    }
}

pub fn stdout(b: &[u8]) {
    write_all(1, b)
}

pub struct Fd(pub i32);
impl fmt::Write for Fd {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        write_all(self.0, s.as_bytes());
        Ok(())
    }
}

#[doc(hidden)]
pub fn _print(fd: i32, args: fmt::Arguments) {
    let _ = fmt::write(&mut Fd(fd), args);
}

#[macro_export]
macro_rules! out { ($($a:tt)*) => { $crate::sys::_print(1, format_args!($($a)*)) } }
#[macro_export]
macro_rules! err { ($($a:tt)*) => { $crate::sys::_print(2, format_args!($($a)*)) } }

pub fn read_line(line: &mut Vec<u8>) -> bool {
    line.clear();
    let mut b = [0u8; 1];
    loop {
        let n = unsafe { read(0, b.as_mut_ptr() as *mut c_void, 1) };
        if n <= 0 {
            return !line.is_empty();
        }
        if b[0] == b'\n' {
            return true;
        }
        line.push(b[0]);
    }
}

pub fn read_stdin_all() -> Vec<u8> {
    let mut v = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = unsafe { read(0, chunk.as_mut_ptr() as *mut c_void, chunk.len()) };
        if n <= 0 {
            return v;
        }
        v.extend_from_slice(&chunk[..n as usize]);
    }
}

pub fn write_file(path: &[u8], data: &[u8]) -> Result<()> {
    unsafe {
        let fd = open(
            path.as_ptr() as *const c_char,
            1 | 0x40 | 0x200,
            0o644 as c_int,
        );
        if fd < 0 {
            return Err(Error::Io("cannot create output file"));
        }
        write_all(fd, data);
        close(fd);
    }
    Ok(())
}

pub fn read_file(path: &[u8]) -> Result<AlignedBuf> {
    unsafe {
        let fd = open(path.as_ptr() as *const c_char, 0);
        if fd < 0 {
            return Err(Error::Io("cannot open file"));
        }
        let size = lseek(fd, 0, 2);
        if size < 0 || lseek(fd, 0, 0) < 0 {
            close(fd);
            return Err(Error::Io("cannot get file size"));
        }
        let size = size as usize;
        let mut buf = match AlignedBuf::new(size) {
            Some(b) => b,
            None => {
                close(fd);
                return Err(Error::Io("out of memory for model file"));
            }
        };
        let dst = buf.as_mut_slice();
        let mut got = 0usize;
        while got < size {
            let n = read(fd, dst.as_mut_ptr().add(got) as *mut c_void, size - got);
            if n <= 0 {
                close(fd);
                return Err(Error::Io("short read"));
            }
            got += n as usize;
        }
        close(fd);
        Ok(buf)
    }
}
