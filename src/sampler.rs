use crate::math::softmax;
use alloc::vec::Vec;

pub struct Sampler {
    pub temperature: f32,
    pub topp: f32,
    state: u64,
    scratch: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler {
            temperature,
            topp,
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            scratch: Vec::new(),
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        for l in logits.iter_mut() {
            *l /= self.temperature;
        }
        softmax(logits);
        let coin = self.next_f32();
        if self.topp <= 0.0 || self.topp >= 1.0 {
            return sample_mult(logits, coin);
        }

        let cutoff = (1.0 - self.topp) / (logits.len() as f32 - 1.0);
        self.scratch.clear();
        for (i, &p) in logits.iter().enumerate() {
            if p >= cutoff {
                self.scratch.push((p, i as u32));
            }
        }
        self.scratch
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        let mut cum = 0.0;
        let mut last = self.scratch.len() - 1;
        for (i, &(p, _)) in self.scratch.iter().enumerate() {
            cum += p;
            if cum > self.topp {
                last = i;
                break;
            }
        }
        let r = coin * cum_upto(&self.scratch, last);
        let mut cdf = 0.0;
        for &(p, id) in &self.scratch[..=last] {
            cdf += p;
            if r < cdf {
                return id;
            }
        }
        self.scratch[last].1
    }
}

fn cum_upto(v: &[(f32, u32)], last: usize) -> f32 {
    v[..=last].iter().map(|e| e.0).sum()
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    let mut bv = x[0];
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

fn sample_mult(p: &[f32], coin: f32) -> u32 {
    let mut cdf = 0.0;
    for (i, &v) in p.iter().enumerate() {
        cdf += v;
        if coin < cdf {
            return i as u32;
        }
    }
    (p.len() - 1) as u32
}
