#[link(name = "m")]
extern "C" {
    fn expf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn powf(x: f32, y: f32) -> f32;
}

#[inline]
pub fn exp(x: f32) -> f32 {
    unsafe { expf(x) }
}
#[inline]
pub fn sqrt(x: f32) -> f32 {
    unsafe { sqrtf(x) }
}
#[inline]
pub fn sin(x: f32) -> f32 {
    unsafe { sinf(x) }
}
#[inline]
pub fn cos(x: f32) -> f32 {
    unsafe { cosf(x) }
}
#[inline]
pub fn pow(x: f32, y: f32) -> f32 {
    unsafe { powf(x, y) }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / sqrt(ss / n as f32 + eps);
    for i in 0..n {
        out[i] = w[i] * (inv * x[i]);
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / sqrt(ss / n as f32 + eps);
    for i in 0..n {
        x[i] = w[i] * (inv * x[i]);
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max = x[0];
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = exp(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let i = c * 8;
        for j in 0..8 {
            acc[j] += a[i + j] * b[i + j];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}
