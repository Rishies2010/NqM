use crate::error::{Error, Result};
use crate::format::Model;
use crate::unicode::{is_letter, is_number, is_other, is_space};
use alloc::vec::Vec;

pub const PRETOK_GPT2: u32 = 0;
pub const PRETOK_LLAMA3: u32 = 1;
pub const PRETOK_QWEN2: u32 = 2;
pub const PRETOK_GPT2_DIGITS: u32 = 3;

pub struct Tokenizer<'a> {
    blob: &'a [u8],

    offs: Vec<u32>,

    sorted: Vec<u32>,
    byte_ids: [u32; 256],

    merges: Vec<(u64, u32, u32)>,
    pretok: u32,
    ignore_merges: bool,
    vocab: usize,
}

fn rd(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

impl<'a> Tokenizer<'a> {
    pub fn new(m: &Model<'a>) -> Result<Tokenizer<'a>> {
        let vocab = m.cfg.vocab_size;
        let s = m.tok_section;
        let lens_end = vocab * 4;
        let blob_bytes = m.tok.blob_bytes;
        let blob_end = lens_end + blob_bytes;
        let merges_off = (blob_end + 3) & !3;
        let merges_end = merges_off + m.tok.n_merges * 12;
        if merges_end > s.len() {
            return Err(Error::Truncated);
        }
        let blob = &s[lens_end..blob_end];
        let mut offs = Vec::with_capacity(vocab + 1);
        let mut acc = 0u32;
        offs.push(0);
        for i in 0..vocab {
            acc = acc
                .checked_add(rd(s, i * 4))
                .ok_or(Error::Invalid("token lengths"))?;
            offs.push(acc);
        }
        if acc as usize != blob_bytes {
            return Err(Error::Invalid("token lengths do not match blob size"));
        }
        let mut sorted: Vec<u32> = (0..vocab as u32)
            .filter(|&i| offs[i as usize + 1] > offs[i as usize])
            .collect();
        sorted.sort_unstable_by(|&a, &b| {
            let ta = &blob[offs[a as usize] as usize..offs[a as usize + 1] as usize];
            let tb = &blob[offs[b as usize] as usize..offs[b as usize + 1] as usize];
            ta.cmp(tb)
        });
        let mut t = Tokenizer {
            blob,
            offs,
            sorted,
            byte_ids: [u32::MAX; 256],
            merges: Vec::with_capacity(m.tok.n_merges),
            pretok: m.tok.pretok,
            ignore_merges: m.tok.ignore_merges,
            vocab,
        };
        for b in 0..256usize {
            let bb = [b as u8];
            t.byte_ids[b] = t
                .lookup(&bb)
                .ok_or(Error::Invalid("tokenizer lacks a single-byte token"))?;
        }
        for r in 0..m.tok.n_merges {
            let o = merges_off + r * 12;
            let (l, rr, mg) = (rd(s, o), rd(s, o + 4), rd(s, o + 8));
            if l as usize >= vocab || rr as usize >= vocab || mg as usize >= vocab {
                return Err(Error::Invalid("merge refers to unknown token"));
            }
            t.merges
                .push((((l as u64) << 32) | rr as u64, r as u32, mg));
        }
        t.merges.sort_unstable_by_key(|e| e.0);
        match t.pretok {
            PRETOK_GPT2 | PRETOK_LLAMA3 | PRETOK_QWEN2 | PRETOK_GPT2_DIGITS => {}
            _ => return Err(Error::Unsupported("pre-tokenizer kind")),
        }
        Ok(t)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        if i >= self.vocab {
            return &[];
        }
        &self.blob[self.offs[i] as usize..self.offs[i + 1] as usize]
    }

    pub fn lookup(&self, bytes: &[u8]) -> Option<u32> {
        let r = self
            .sorted
            .binary_search_by(|&id| self.token_bytes(id).cmp(bytes));
        r.ok().map(|i| self.sorted[i])
    }

    fn merge_for(&self, l: u32, r: u32) -> Option<(u32, u32)> {
        let key = ((l as u64) << 32) | r as u64;
        self.merges
            .binary_search_by_key(&key, |e| e.0)
            .ok()
            .map(|i| (self.merges[i].1, self.merges[i].2))
    }

    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>, scratch: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        if self.ignore_merges {
            if let Some(id) = self.lookup(piece) {
                out.push(id);
                return;
            }
        }
        scratch.clear();
        scratch.extend(piece.iter().map(|&b| self.byte_ids[b as usize]));
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            let mut best_tok = 0;
            for i in 0..scratch.len().saturating_sub(1) {
                if let Some((rank, tok)) = self.merge_for(scratch[i], scratch[i + 1]) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i;
                        best_tok = tok;
                    }
                }
            }
            if best_i == usize::MAX {
                break;
            }
            scratch[best_i] = best_tok;
            scratch.remove(best_i + 1);
        }
        out.extend_from_slice(scratch);
    }

    pub fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let mut chars: Vec<char> = Vec::with_capacity(text.len());
        let mut boff: Vec<usize> = Vec::with_capacity(text.len() + 1);
        for (o, c) in text.char_indices() {
            chars.push(c);
            boff.push(o);
        }
        boff.push(text.len());
        let bytes = text.as_bytes();
        let mut scratch = Vec::new();
        pretokenize(&chars, self.pretok, &mut |a, b| {
            self.bpe(&bytes[boff[a]..boff[b]], out, &mut scratch);
        });
    }
}

pub fn pretokenize(ch: &[char], kind: u32, f: &mut dyn FnMut(usize, usize)) {
    match kind {
        PRETOK_LLAMA3 | PRETOK_QWEN2 => {
            let digits = if kind == PRETOK_LLAMA3 { 3 } else { 1 };
            let mut i = 0;
            while i < ch.len() {
                let e = piece_llama(ch, i, digits);
                f(i, e);
                i = e;
            }
        }
        PRETOK_GPT2 => {
            let mut i = 0;
            while i < ch.len() {
                let e = piece_gpt2(ch, i, ch.len());
                f(i, e);
                i = e;
            }
        }
        _ => {
            let n = ch.len();
            let mut i = 0;
            while i < n {
                if is_number(ch[i]) {
                    f(i, i + 1);
                    i += 1;
                    continue;
                }
                let mut j = i;
                while j < n && !is_number(ch[j]) {
                    j += 1;
                }
                let mut k = i;
                while k < j {
                    let e = piece_gpt2(ch, k, j);
                    f(k, e);
                    k = e;
                }
                i = j;
            }
        }
    }
}

fn contraction(ch: &[char], i: usize, end: usize, case_insensitive: bool) -> Option<usize> {
    let get = |k: usize| -> Option<char> {
        if i + k < end {
            let c = ch[i + k];
            Some(if case_insensitive {
                c.to_ascii_lowercase()
            } else {
                c
            })
        } else {
            None
        }
    };
    match get(1)? {
        's' | 't' | 'm' | 'd' => Some(2),
        'r' if get(2) == Some('e') => Some(3),
        'v' if get(2) == Some('e') => Some(3),
        'l' if get(2) == Some('l') => Some(3),
        _ => None,
    }
}

fn space_run(ch: &[char], i: usize, end: usize) -> usize {
    let mut e = i;
    while e < end && is_space(ch[e]) {
        e += 1;
    }
    if e == end {
        return e;
    }
    if e - i >= 2 {
        return e - 1;
    }
    e
}

fn piece_llama(ch: &[char], i: usize, digits: usize) -> usize {
    let n = ch.len();
    let c = ch[i];
    if c == '\'' {
        if let Some(l) = contraction(ch, i, n, true) {
            return i + l;
        }
    }

    if is_letter(c) {
        let mut j = i + 1;
        while j < n && is_letter(ch[j]) {
            j += 1;
        }
        return j;
    }
    if c != '\r' && c != '\n' && !is_number(c) && i + 1 < n && is_letter(ch[i + 1]) {
        let mut j = i + 2;
        while j < n && is_letter(ch[j]) {
            j += 1;
        }
        return j;
    }

    if is_number(c) {
        let mut j = i + 1;
        while j < n && j - i < digits && is_number(ch[j]) {
            j += 1;
        }
        return j;
    }

    {
        let mut j = if c == ' ' { i + 1 } else { i };
        if j < n && is_other(ch[j]) {
            while j < n && is_other(ch[j]) {
                j += 1;
            }
            while j < n && (ch[j] == '\r' || ch[j] == '\n') {
                j += 1;
            }
            return j;
        }
    }

    let mut e = i;
    while e < n && is_space(ch[e]) {
        e += 1;
    }
    let mut k = e;
    while k > i {
        k -= 1;
        if ch[k] == '\r' || ch[k] == '\n' {
            return k + 1;
        }
    }
    space_run(ch, i, n)
}

fn piece_gpt2(ch: &[char], i: usize, end: usize) -> usize {
    let c = ch[i];
    if c == '\'' {
        if let Some(l) = contraction(ch, i, end, false) {
            return i + l;
        }
    }
    let j0 = if c == ' ' { i + 1 } else { i };
    if j0 < end {
        let d = ch[j0];
        let class: fn(char) -> bool = if is_letter(d) {
            is_letter
        } else if is_number(d) {
            is_number
        } else if is_other(d) {
            is_other
        } else {
            |_| false
        };
        if class(d) {
            let mut j = j0 + 1;
            while j < end && class(ch[j]) {
                j += 1;
            }
            return j;
        }
    }
    space_run(ch, i, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec::Vec;

    fn split(s: &str, kind: u32) -> Vec<String> {
        let ch: Vec<char> = s.chars().collect();
        let mut out = Vec::new();
        pretokenize(&ch, kind, &mut |a, b| {
            out.push(ch[a..b].iter().collect::<String>())
        });
        out
    }

    #[test]
    fn llama3_basic() {
        assert_eq!(
            split("Hello, world!", PRETOK_LLAMA3),
            ["Hello", ",", " world", "!"]
        );
        assert_eq!(
            split("it's 12345", PRETOK_LLAMA3),
            ["it", "'s", " ", "123", "45"]
        );
        assert_eq!(split("a  b", PRETOK_LLAMA3), ["a", " ", " b"]);
        assert_eq!(split("x\n\ny", PRETOK_LLAMA3), ["x", "\n\n", "y"]);
        assert_eq!(split("12345", PRETOK_QWEN2), ["1", "2", "3", "4", "5"]);
    }

    #[test]
    fn gpt2_basic() {
        assert_eq!(
            split("Hello, world!", PRETOK_GPT2),
            ["Hello", ",", " world", "!"]
        );
        assert_eq!(split("don't 12", PRETOK_GPT2), ["don", "'t", " 12"]);
        assert_eq!(split("a1", PRETOK_GPT2_DIGITS), ["a", "1"]);
    }
}
