use crate::format::TokInfo;
use crate::tokenizer::Tokenizer;
use alloc::format;
use alloc::vec::Vec;

pub const CHAT_NONE: u32 = 0;
pub const CHAT_LLAMA3: u32 = 1;
pub const CHAT_CHATML: u32 = 2;

fn sp(out: &mut Vec<u32>, id: i32) {
    if id >= 0 {
        out.push(id as u32);
    }
}

pub fn user_turn(
    t: &Tokenizer,
    info: &TokInfo,
    system: Option<&str>,
    user: &str,
    first: bool,
    out: &mut Vec<u32>,
) -> bool {
    match info.chat_kind {
        CHAT_LLAMA3 => {
            let (start, end, eot) = (info.tpl[0], info.tpl[1], info.tpl[2]);
            if first {
                sp(out, info.bos);
                if let Some(s) = system {
                    sp(out, start);
                    t.encode("system", out);
                    sp(out, end);
                    t.encode(&format!("\n\n{s}"), out);
                    sp(out, eot);
                }
            }
            sp(out, start);
            t.encode("user", out);
            sp(out, end);
            t.encode(&format!("\n\n{user}"), out);
            sp(out, eot);
            sp(out, start);
            t.encode("assistant", out);
            sp(out, end);
            t.encode("\n\n", out);
            true
        }
        CHAT_CHATML => {
            let (start, end) = (info.tpl[0], info.tpl[1]);
            if first {
                if let Some(s) = system {
                    sp(out, start);
                    t.encode(&format!("system\n{s}"), out);
                    sp(out, end);
                    t.encode("\n", out);
                }
            } else {
                t.encode("\n", out);
            }
            sp(out, start);
            t.encode(&format!("user\n{user}"), out);
            sp(out, end);
            t.encode("\n", out);
            sp(out, start);
            t.encode("assistant\n", out);
            true
        }
        _ => false,
    }
}
