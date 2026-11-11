use crate::error::{Error, Result};
use crate::quant::{as_f32, QMat, QType};
use alloc::vec::Vec;

pub const MAGIC: u32 = 0x314D_514E;
pub const HEADER_BYTES: usize = 512;

const W_VERSION: usize = 1;
const W_HEADER_BYTES: usize = 2;
const W_FLAGS: usize = 3;
const W_DIM: usize = 4;
const W_HIDDEN: usize = 5;
const W_LAYERS: usize = 6;
const W_HEADS: usize = 7;
const W_KV_HEADS: usize = 8;
const W_HEAD_DIM: usize = 9;
const W_VOCAB: usize = 10;
const W_MAXSEQ: usize = 11;
const W_ROPE_THETA: usize = 12;
const W_NORM_EPS: usize = 13;
const W_ROPE_KIND: usize = 14;
const W_ROPE_FACTOR: usize = 15;
const W_ROPE_LOW: usize = 16;
const W_ROPE_HIGH: usize = 17;
const W_ROPE_ORIG: usize = 18;
const W_QBODY: usize = 19;
const W_QEMBED: usize = 20;
const W_GS_BODY: usize = 21;
const W_GS_EMBED: usize = 22;
const W_PRETOK: usize = 23;
const W_TOKFLAGS: usize = 24;
const W_NMERGES: usize = 25;
const W_BLOB: usize = 26;
const W_BOS: usize = 27;
const W_STOP0: usize = 28;
const W_CHAT: usize = 32;
const W_TPL0: usize = 33;
const W_OFF_TOK: usize = 38;
const W_OFF_W: usize = 40;
const W_FILESIZE: usize = 42;
const W_NAME: usize = 44;

pub const FLAG_TIED: u32 = 1;
pub const FLAG_BIAS: u32 = 2;
pub const FLAG_QKNORM: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    Llama3 {
        factor: f32,
        low_freq: f32,
        high_freq: f32,
        orig_ctx: f32,
    },
}

#[derive(Clone, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub rope: RopeScaling,
    pub tied: bool,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub quant_body: QType,
    pub quant_embed: QType,
    pub gs_body: usize,
    pub gs_embed: usize,
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

#[derive(Clone, Debug)]
pub struct TokInfo {
    pub pretok: u32,
    pub ignore_merges: bool,
    pub n_merges: usize,
    pub blob_bytes: usize,
    pub bos: i32,
    pub stops: [i32; 4],
    pub chat_kind: u32,
    pub tpl: [i32; 4],
}

pub struct Layer<'a> {
    pub attn_norm: &'a [f32],
    pub wq: QMat<'a>,
    pub wk: QMat<'a>,
    pub wv: QMat<'a>,
    pub wo: QMat<'a>,
    pub bq: Option<&'a [f32]>,
    pub bk: Option<&'a [f32]>,
    pub bv: Option<&'a [f32]>,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub ffn_norm: &'a [f32],
    pub w_gate: QMat<'a>,
    pub w_up: QMat<'a>,
    pub w_down: QMat<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    pub tok: TokInfo,
    pub name: [u8; 32],
    pub file_bytes: usize,
    pub(crate) tok_section: &'a [u8],
    pub embed: QMat<'a>,
    pub layers: Vec<Layer<'a>>,
    pub final_norm: &'a [f32],

    pub lm_head: QMat<'a>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated)?;
        if end > self.buf.len() {
            return Err(Error::Truncated);
        }
        let s = &self.buf[self.pos..end];
        self.pos = (end + 63) & !63;
        Ok(s)
    }
    fn vec(&mut self, n: usize) -> Result<&'a [f32]> {
        Ok(as_f32(self.take(n * 4)?))
    }
    fn matrix(&mut self, rows: usize, cols: usize, qt: QType, gs: usize) -> Result<QMat<'a>> {
        if qt != QType::F32 && (gs == 0 || cols % gs != 0 || gs % 8 != 0) {
            return Err(Error::Invalid(
                "group size must divide columns and be a multiple of 8",
            ));
        }
        let data_bytes = match qt {
            QType::F32 => rows * cols * 4,
            QType::Q8 => rows * cols,
            QType::Q4 => rows * cols / 2,
        };
        let data = self.take(data_bytes)?;
        let scales: &'a [f32] = if qt == QType::F32 {
            &[]
        } else {
            as_f32(self.take(rows * (cols / gs) * 4)?)
        };
        Ok(QMat {
            qt,
            rows,
            cols,
            gs,
            data,
            scales,
        })
    }
}

fn word(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
}
fn wordf(b: &[u8], i: usize) -> f32 {
    f32::from_bits(word(b, i))
}
fn word64(b: &[u8], i: usize) -> u64 {
    word(b, i) as u64 | ((word(b, i + 1) as u64) << 32)
}

impl<'a> Model<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Model<'a>> {
        if bytes.as_ptr() as usize % 64 != 0 {
            return Err(Error::Misaligned);
        }
        if bytes.len() < HEADER_BYTES {
            return Err(Error::Truncated);
        }
        if word(bytes, 0) != MAGIC {
            return Err(Error::BadMagic);
        }
        let ver = word(bytes, W_VERSION);
        if ver != 1 {
            return Err(Error::BadVersion(ver));
        }
        if word(bytes, W_HEADER_BYTES) as usize != HEADER_BYTES {
            return Err(Error::Invalid("header size"));
        }
        let file_size = word64(bytes, W_FILESIZE) as usize;
        if file_size != bytes.len() {
            return Err(Error::Truncated);
        }
        let flags = word(bytes, W_FLAGS);
        let quant_body =
            QType::from_u32(word(bytes, W_QBODY)).ok_or(Error::Unsupported("body quant type"))?;
        let quant_embed =
            QType::from_u32(word(bytes, W_QEMBED)).ok_or(Error::Unsupported("embed quant type"))?;
        if quant_embed == QType::Q4 {
            return Err(Error::Unsupported("q4 embeddings"));
        }
        let rope = match word(bytes, W_ROPE_KIND) {
            0 => RopeScaling::None,
            1 => RopeScaling::Llama3 {
                factor: wordf(bytes, W_ROPE_FACTOR),
                low_freq: wordf(bytes, W_ROPE_LOW),
                high_freq: wordf(bytes, W_ROPE_HIGH),
                orig_ctx: word(bytes, W_ROPE_ORIG) as f32,
            },
            _ => return Err(Error::Unsupported("rope scaling kind")),
        };
        let cfg = Config {
            dim: word(bytes, W_DIM) as usize,
            hidden_dim: word(bytes, W_HIDDEN) as usize,
            n_layers: word(bytes, W_LAYERS) as usize,
            n_heads: word(bytes, W_HEADS) as usize,
            n_kv_heads: word(bytes, W_KV_HEADS) as usize,
            head_dim: word(bytes, W_HEAD_DIM) as usize,
            vocab_size: word(bytes, W_VOCAB) as usize,
            max_seq_len: word(bytes, W_MAXSEQ) as usize,
            rope_theta: wordf(bytes, W_ROPE_THETA),
            norm_eps: wordf(bytes, W_NORM_EPS),
            rope,
            tied: flags & FLAG_TIED != 0,
            qkv_bias: flags & FLAG_BIAS != 0,
            qk_norm: flags & FLAG_QKNORM != 0,
            quant_body,
            quant_embed,
            gs_body: word(bytes, W_GS_BODY) as usize,
            gs_embed: word(bytes, W_GS_EMBED) as usize,
        };
        if cfg.n_heads == 0 || cfg.n_kv_heads == 0 || cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(Error::Invalid("heads / kv heads"));
        }
        if cfg.head_dim == 0
            || cfg.head_dim % 2 != 0
            || cfg.dim == 0
            || cfg.n_layers == 0
            || cfg.vocab_size == 0
        {
            return Err(Error::Invalid("dimensions"));
        }
        let mut stops = [-1i32; 4];
        let mut tpl = [-1i32; 4];
        for i in 0..4 {
            stops[i] = word(bytes, W_STOP0 + i) as i32;
            tpl[i] = word(bytes, W_TPL0 + i) as i32;
        }
        let tok = TokInfo {
            pretok: word(bytes, W_PRETOK),
            ignore_merges: word(bytes, W_TOKFLAGS) & 1 != 0,
            n_merges: word(bytes, W_NMERGES) as usize,
            blob_bytes: word(bytes, W_BLOB) as usize,
            bos: word(bytes, W_BOS) as i32,
            stops,
            chat_kind: word(bytes, W_CHAT),
            tpl,
        };
        let mut name = [0u8; 32];
        name.copy_from_slice(&bytes[W_NAME * 4..W_NAME * 4 + 32]);

        let off_tok = word64(bytes, W_OFF_TOK) as usize;
        let off_w = word64(bytes, W_OFF_W) as usize;
        if off_tok % 64 != 0 || off_w % 64 != 0 || off_tok > off_w || off_w > bytes.len() {
            return Err(Error::Invalid("section offsets"));
        }
        let tok_section = &bytes[off_tok..off_w];

        let mut c = Cursor {
            buf: bytes,
            pos: off_w,
        };
        let (d, hd, h) = (cfg.dim, cfg.hidden_dim, cfg.head_dim);
        let (qd, kd) = (cfg.q_dim(), cfg.kv_dim());
        let (qb, gb) = (cfg.quant_body, cfg.gs_body);
        let embed = c.matrix(cfg.vocab_size, d, quant_embed, cfg.gs_embed)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let attn_norm = c.vec(d)?;
            let wq = c.matrix(qd, d, qb, gb)?;
            let wk = c.matrix(kd, d, qb, gb)?;
            let wv = c.matrix(kd, d, qb, gb)?;
            let wo = c.matrix(d, qd, qb, gb)?;
            let (mut bq, mut bk, mut bv) = (None, None, None);
            if cfg.qkv_bias {
                bq = Some(c.vec(qd)?);
                bk = Some(c.vec(kd)?);
                bv = Some(c.vec(kd)?);
            }
            let (mut q_norm, mut k_norm) = (None, None);
            if cfg.qk_norm {
                q_norm = Some(c.vec(h)?);
                k_norm = Some(c.vec(h)?);
            }
            let ffn_norm = c.vec(d)?;
            let w_gate = c.matrix(hd, d, qb, gb)?;
            let w_up = c.matrix(hd, d, qb, gb)?;
            let w_down = c.matrix(d, hd, qb, gb)?;
            layers.push(Layer {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                bq,
                bk,
                bv,
                q_norm,
                k_norm,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }
        let final_norm = c.vec(d)?;
        let lm_head = if cfg.tied {
            embed
        } else {
            c.matrix(cfg.vocab_size, d, quant_embed, cfg.gs_embed)?
        };
        Ok(Model {
            cfg,
            tok,
            name,
            file_bytes: bytes.len(),
            tok_section,
            embed,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(32);
        core::str::from_utf8(&self.name[..end]).unwrap_or("?")
    }
}
