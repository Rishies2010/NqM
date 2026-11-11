#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QType {
    F32 = 0,
    Q8 = 1,
    Q4 = 2,
}

impl QType {
    pub fn from_u32(v: u32) -> Option<QType> {
        match v {
            0 => Some(QType::F32),
            1 => Some(QType::Q8),
            2 => Some(QType::Q4),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct QMat<'a> {
    pub qt: QType,
    pub rows: usize,
    pub cols: usize,
    pub gs: usize,
    pub(crate) data: &'a [u8],
    pub(crate) scales: &'a [f32],
}

#[inline]
pub(crate) fn as_f32(b: &[u8]) -> &[f32] {
    debug_assert!(b.as_ptr() as usize % 4 == 0 && b.len() % 4 == 0);
    unsafe { core::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}

impl<'a> QMat<'a> {
    pub fn matvec(&self, out: &mut [f32], x: &[f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert!(out.len() >= self.rows);
        match self.qt {
            QType::F32 => self.matvec_f32(out, x),
            QType::Q8 => self.matvec_q8(out, x),
            QType::Q4 => self.matvec_q4(out, x),
        }
    }

    fn matvec_f32(&self, out: &mut [f32], x: &[f32]) {
        let w = as_f32(self.data);
        let cols = self.cols;
        for r in 0..self.rows {
            out[r] = crate::math::dot(&w[r * cols..(r + 1) * cols], x);
        }
    }

    fn matvec_q8(&self, out: &mut [f32], x: &[f32]) {
        let cols = self.cols;
        let gs = self.gs;
        let ng = cols / gs;
        for r in 0..self.rows {
            let w = &self.data[r * cols..(r + 1) * cols];
            let sc = &self.scales[r * ng..(r + 1) * ng];
            let mut total = 0.0f32;
            for g in 0..ng {
                let wg = &w[g * gs..(g + 1) * gs];
                let xg = &x[g * gs..(g + 1) * gs];
                let mut acc = [0.0f32; 8];
                for (wc, xc) in wg.chunks_exact(8).zip(xg.chunks_exact(8)) {
                    for j in 0..8 {
                        acc[j] += (wc[j] as i8) as f32 * xc[j];
                    }
                }
                let s = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                    + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
                total += s * sc[g];
            }
            out[r] = total;
        }
    }

    fn matvec_q4(&self, out: &mut [f32], x: &[f32]) {
        let cols = self.cols;
        let gs = self.gs;
        let ng = cols / gs;
        let rb = cols / 2;
        let gb = gs / 2;
        for r in 0..self.rows {
            let w = &self.data[r * rb..(r + 1) * rb];
            let sc = &self.scales[r * ng..(r + 1) * ng];
            let mut total = 0.0f32;
            for g in 0..ng {
                let wg = &w[g * gb..(g + 1) * gb];
                let xg = &x[g * gs..(g + 1) * gs];
                let mut acc = [0.0f32; 8];
                for (wc, xc) in wg.chunks_exact(4).zip(xg.chunks_exact(8)) {
                    for k in 0..4 {
                        let b = wc[k];
                        acc[2 * k] += ((b & 0x0F) as i32 - 8) as f32 * xc[2 * k];
                        acc[2 * k + 1] += ((b >> 4) as i32 - 8) as f32 * xc[2 * k + 1];
                    }
                }
                let s = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                    + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
                total += s * sc[g];
            }
            out[r] = total;
        }
    }

    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        let cols = self.cols;
        match self.qt {
            QType::F32 => out[..cols].copy_from_slice(&as_f32(self.data)[r * cols..(r + 1) * cols]),
            QType::Q8 => {
                let ng = cols / self.gs;
                let w = &self.data[r * cols..(r + 1) * cols];
                let sc = &self.scales[r * ng..(r + 1) * ng];
                for c in 0..cols {
                    out[c] = (w[c] as i8) as f32 * sc[c / self.gs];
                }
            }
            QType::Q4 => {
                let ng = cols / self.gs;
                let w = &self.data[r * (cols / 2)..(r + 1) * (cols / 2)];
                let sc = &self.scales[r * ng..(r + 1) * ng];
                for c in 0..cols {
                    let b = w[c / 2];
                    let q = if c & 1 == 0 { b & 0x0F } else { b >> 4 } as i32 - 8;
                    out[c] = q as f32 * sc[c / self.gs];
                }
            }
        }
    }
}
