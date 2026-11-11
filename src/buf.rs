use alloc::alloc::{alloc, dealloc, Layout};

pub const ALIGN: usize = 64;

pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuf {
    pub fn new(len: usize) -> Option<Self> {
        let layout = Layout::from_size_align(len.max(1), ALIGN).ok()?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return None;
        }

        unsafe { core::ptr::write_bytes(ptr, 0, len) };
        Some(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len.max(1), ALIGN).unwrap();

        unsafe { dealloc(self.ptr, layout) };
    }
}
