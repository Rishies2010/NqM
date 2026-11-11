use crate::format::{Config, Model, RopeScaling};
use crate::math::{dot, pow, rmsnorm, rmsnorm_inplace, silu, softmax};
use alloc::vec;
use alloc::vec::Vec;

pub struct State {
    pub ctx: usize,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    attn_out: Vec<f32>,
    inv_freq: Vec<f32>,
    cosv: Vec<f32>,
    sinv: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
    pub logits: Vec<f32>,
}

fn rope_inv_freq(cfg: &Config) -> Vec<f32> {
    let half = cfg.head_dim / 2;
    let mut inv = vec![0.0f32; half];
    for i in 0..half {
        let e = (2 * i) as f32 / cfg.head_dim as f32;
        inv[i] = 1.0 / pow(cfg.rope_theta, e);
    }
    if let RopeScaling::Llama3 {
        factor,
        low_freq,
        high_freq,
        orig_ctx,
    } = cfg.rope
    {
        let low_wavelen = orig_ctx / low_freq;
        let high_wavelen = orig_ctx / high_freq;
        let two_pi = 2.0 * core::f32::consts::PI;
        for f in inv.iter_mut() {
            let wavelen = two_pi / *f;
            if wavelen < high_wavelen {
            } else if wavelen > low_wavelen {
                *f /= factor;
            } else {
                let smooth = (orig_ctx / wavelen - low_freq) / (high_freq - low_freq);
                *f = (1.0 - smooth) * (*f / factor) + smooth * *f;
            }
        }
    }
    inv
}

impl State {
    pub fn new(cfg: &Config, ctx: usize) -> State {
        let (qd, kd) = (cfg.q_dim(), cfg.kv_dim());
        State {
            ctx,
            x: vec![0.0; cfg.dim],
            xb: vec![0.0; cfg.dim],
            xb2: vec![0.0; cfg.dim],
            hb: vec![0.0; cfg.hidden_dim],
            hb2: vec![0.0; cfg.hidden_dim],
            q: vec![0.0; qd],
            k: vec![0.0; kd],
            v: vec![0.0; kd],
            att: vec![0.0; cfg.n_heads * ctx],
            attn_out: vec![0.0; qd],
            inv_freq: rope_inv_freq(cfg),
            cosv: vec![0.0; cfg.head_dim / 2],
            sinv: vec![0.0; cfg.head_dim / 2],
            kc: vec![0.0; cfg.n_layers * ctx * kd],
            vc: vec![0.0; cfg.n_layers * ctx * kd],
            logits: vec![0.0; cfg.vocab_size],
        }
    }

    pub fn kv_bytes(cfg: &Config, ctx: usize) -> usize {
        2 * cfg.n_layers * ctx * cfg.kv_dim() * 4
    }

    pub fn forward(
        &mut self,
        m: &Model,
        token: u32,
        pos: usize,
        want_logits: bool,
    ) -> Option<&[f32]> {
        let cfg = &m.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let (nh, nkv) = (cfg.n_heads, cfg.n_kv_heads);
        let kd = cfg.kv_dim();
        let group = nh / nkv;
        let ctx = self.ctx;
        assert!(pos < ctx, "position beyond context");
        assert!((token as usize) < cfg.vocab_size, "token id out of range");
        let scale = 1.0 / crate::math::sqrt(hd as f32);

        m.embed.row_into(token as usize, &mut self.x);

        for i in 0..hd / 2 {
            let ang = pos as f32 * self.inv_freq[i];
            self.cosv[i] = crate::math::cos(ang);
            self.sinv[i] = crate::math::sin(ang);
        }

        for (l, lw) in m.layers.iter().enumerate() {
            rmsnorm(&mut self.xb, &self.x, lw.attn_norm, cfg.norm_eps);
            lw.wq.matvec(&mut self.q, &self.xb);
            lw.wk.matvec(&mut self.k, &self.xb);
            lw.wv.matvec(&mut self.v, &self.xb);
            if let (Some(bq), Some(bk), Some(bv)) = (lw.bq, lw.bk, lw.bv) {
                for i in 0..self.q.len() {
                    self.q[i] += bq[i];
                }
                for i in 0..kd {
                    self.k[i] += bk[i];
                    self.v[i] += bv[i];
                }
            }
            if let (Some(qn), Some(kn)) = (lw.q_norm, lw.k_norm) {
                for h in 0..nh {
                    rmsnorm_inplace(&mut self.q[h * hd..(h + 1) * hd], qn, cfg.norm_eps);
                }
                for h in 0..nkv {
                    rmsnorm_inplace(&mut self.k[h * hd..(h + 1) * hd], kn, cfg.norm_eps);
                }
            }

            {
                let (cosv, sinv) = (&self.cosv, &self.sinv);
                let half = hd / 2;
                for h in 0..nh {
                    let o = h * hd;
                    for i in 0..half {
                        let (a, b) = (self.q[o + i], self.q[o + i + half]);
                        self.q[o + i] = a * cosv[i] - b * sinv[i];
                        self.q[o + i + half] = b * cosv[i] + a * sinv[i];
                    }
                }
                for h in 0..nkv {
                    let o = h * hd;
                    for i in 0..half {
                        let (a, b) = (self.k[o + i], self.k[o + i + half]);
                        self.k[o + i] = a * cosv[i] - b * sinv[i];
                        self.k[o + i + half] = b * cosv[i] + a * sinv[i];
                    }
                }
            }

            let base = (l * ctx + pos) * kd;
            self.kc[base..base + kd].copy_from_slice(&self.k);
            self.vc[base..base + kd].copy_from_slice(&self.v);

            for h in 0..nh {
                let kvh = h / group;
                let qh = &self.q[h * hd..(h + 1) * hd];
                let att = &mut self.att[h * ctx..h * ctx + pos + 1];
                for t in 0..=pos {
                    let ko = (l * ctx + t) * kd + kvh * hd;
                    att[t] = dot(qh, &self.kc[ko..ko + hd]) * scale;
                }
                softmax(att);
                let out = &mut self.attn_out[h * hd..(h + 1) * hd];
                for o in out.iter_mut() {
                    *o = 0.0;
                }
                for t in 0..=pos {
                    let vo = (l * ctx + t) * kd + kvh * hd;
                    let a = att[t];
                    let vt = &self.vc[vo..vo + hd];
                    for i in 0..hd {
                        out[i] += a * vt[i];
                    }
                }
            }
            lw.wo.matvec(&mut self.xb2, &self.attn_out);
            for i in 0..dim {
                self.x[i] += self.xb2[i];
            }

            rmsnorm(&mut self.xb, &self.x, lw.ffn_norm, cfg.norm_eps);
            lw.w_gate.matvec(&mut self.hb, &self.xb);
            lw.w_up.matvec(&mut self.hb2, &self.xb);
            for i in 0..cfg.hidden_dim {
                self.hb[i] = silu(self.hb[i]) * self.hb2[i];
            }
            lw.w_down.matvec(&mut self.xb2, &self.hb);
            for i in 0..dim {
                self.x[i] += self.xb2[i];
            }
        }

        if !want_logits {
            return None;
        }
        rmsnorm(&mut self.xb, &self.x, m.final_norm, cfg.norm_eps);
        m.lm_head.matvec(&mut self.logits, &self.xb);
        Some(&self.logits)
    }
}
