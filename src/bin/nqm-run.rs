//! nqm-run: run an .nqm model.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use core::ffi::c_char;
use nqm::buf::AlignedBuf;
use nqm::format::Model;
use nqm::model::State;
use nqm::sampler::Sampler;
use nqm::tokenizer::Tokenizer;
use nqm::{chat, err, out, sys};

#[global_allocator]
static ALLOC: sys::LibcAlloc = sys::LibcAlloc;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    err!("\nnqm-run panic: {}\n", info);
    unsafe { sys::abort() }
}

#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
#[no_mangle]
pub extern "C" fn _Unwind_Resume() -> ! {
    unsafe { sys::abort() }
}

const USAGE: &str = "\
nqm-run MODEL.nqm [options]

  -p TEXT           raw completion of TEXT (BOS added if the model requests)
  --ask TEXT        one chat turn (applies model's chat template), then exit
  --chat            interactive chat, one message per stdin line (/quit to leave)
  --system TEXT     system prompt for chat modes
  -n N              max new tokens (default 256)
  -c N              context length (default min(model max, 2048))
  -t F              temperature, 0 = greedy (default 0.7)
  --topp F          nucleus sampling (default 0.9)
  --seed N          RNG seed (default: clock)

  --info            print model info and exit
  --encode TEXT     print token ids of TEXT (no BOS, specials not parsed) and exit
  --encode-stdin    like --encode; records on stdin separated by NUL, one line out each
  --decode IDS      print the bytes of comma-separated token ids and exit
  --chat-ids TEXT   print the token ids of a first chat turn (template + TEXT, honours --system) and exit
  --ids IDS         use comma-separated token ids as the prompt
  --dump-logits F   write f32 logits for every prompt position to file F
";

fn cstr(p: *const c_char) -> &'static [u8] {
    unsafe {
        let mut n = 0;
        while *p.add(n) != 0 {
            n += 1;
        }
        core::slice::from_raw_parts(p as *const u8, n + 1)
    }
}

fn text(b: &'static [u8]) -> &'static str {
    core::str::from_utf8(&b[..b.len() - 1]).unwrap_or("")
}

struct Args {
    model: &'static [u8],
    prompt: Option<&'static str>,
    ask: Option<&'static str>,
    chat: bool,
    system: Option<&'static str>,
    n: usize,
    ctx: Option<usize>,
    temp: f32,
    topp: f32,
    seed: Option<u64>,
    info: bool,
    encode: Option<&'static str>,
    encode_stdin: bool,
    decode: Option<&'static str>,
    chat_ids: Option<&'static str>,
    ids: Option<&'static str>,
    dump: Option<&'static [u8]>,
}

fn parse_args(a: &[&'static [u8]]) -> Option<Args> {
    let mut r = Args {
        model: b"\0",
        prompt: None,
        ask: None,
        chat: false,
        system: None,
        n: 256,
        ctx: None,
        temp: 0.7,
        topp: 0.9,
        seed: None,
        info: false,
        encode: None,
        encode_stdin: false,
        decode: None,
        chat_ids: None,
        ids: None,
        dump: None,
    };
    let mut have_model = false;
    let mut i = 1;
    macro_rules! val {
        () => {{
            i += 1;
            if i >= a.len() {
                err!("missing value for {}\n", text(a[i - 1]));
                return None;
            }
            text(a[i])
        }};
    }
    macro_rules! num {
        ($t:ty) => {{
            match val!().parse::<$t>() {
                Ok(v) => v,
                Err(_) => {
                    err!("bad number for {}\n", text(a[i - 1]));
                    return None;
                }
            }
        }};
    }
    while i < a.len() {
        match text(a[i]) {
            "-p" => r.prompt = Some(val!()),
            "--ask" => r.ask = Some(val!()),
            "--chat" => r.chat = true,
            "--system" => r.system = Some(val!()),
            "-n" => r.n = num!(usize),
            "-c" => r.ctx = Some(num!(usize)),
            "-t" => r.temp = num!(f32),
            "--topp" => r.topp = num!(f32),
            "--seed" => r.seed = Some(num!(u64)),
            "--info" => r.info = true,
            "--encode" => r.encode = Some(val!()),
            "--encode-stdin" => r.encode_stdin = true,
            "--decode" => r.decode = Some(val!()),
            "--chat-ids" => r.chat_ids = Some(val!()),
            "--ids" => r.ids = Some(val!()),
            "--dump-logits" => {
                i += 1;
                if i >= a.len() {
                    err!("missing value for --dump-logits\n");
                    return None;
                }
                r.dump = Some(a[i]);
            }
            "-h" | "--help" => return None,
            s if s.starts_with('-') && s.len() > 1 => {
                err!("unknown option {}\n", s);
                return None;
            }
            _ => {
                r.model = a[i];
                have_model = true;
            }
        }
        i += 1;
    }
    if have_model {
        Some(r)
    } else {
        None
    }
}

fn parse_ids(s: &str) -> Option<Vec<u32>> {
    let mut v = Vec::new();
    for p in s.split(',') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        v.push(p.parse::<u32>().ok()?);
    }
    Some(v)
}

fn print_ids(ids: &[u32]) {
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            out!(" ");
        }
        out!("{}", id);
    }
    out!("\n");
}

fn feed(m: &Model, st: &mut State, ids: &[u32], pos: &mut usize, dump: &mut Option<Vec<f32>>) {
    for (k, &t) in ids.iter().enumerate() {
        let want = k + 1 == ids.len() || dump.is_some();
        st.forward(m, t, *pos, want);
        if let Some(d) = dump {
            d.extend_from_slice(&st.logits);
        }
        *pos += 1;
    }
}

fn generate(
    m: &Model,
    tok: &Tokenizer,
    st: &mut State,
    sampler: &mut Sampler,
    pos: &mut usize,
    max_new: usize,
) -> usize {
    let mut count = 0;
    while count < max_new {
        let next = sampler.sample(&mut st.logits);
        let is_stop = m.tok.stops.iter().any(|&s| s >= 0 && s as u32 == next);
        if *pos >= st.ctx {
            err!("\n[context full]\n");
            break;
        }
        if is_stop {
            st.forward(m, next, *pos, false);
            *pos += 1;
            break;
        }
        sys::stdout(tok.token_bytes(next));
        st.forward(m, next, *pos, true);
        *pos += 1;
        count += 1;
    }
    count
}

fn run(a: &[&'static [u8]]) -> i32 {
    let args = match parse_args(a) {
        Some(x) => x,
        None => {
            err!("{}", USAGE);
            return 2;
        }
    };
    let buf: AlignedBuf = match sys::read_file(args.model) {
        Ok(b) => b,
        Err(e) => {
            err!("{}: {}\n", text(args.model), e);
            return 1;
        }
    };
    let model = match Model::parse(buf.as_slice()) {
        Ok(m) => m,
        Err(e) => {
            err!("{}: {}\n", text(args.model), e);
            return 1;
        }
    };
    let c = &model.cfg;
    err!(
        "nqm: {} | {} layers, dim {}, heads {}/{} (hd {}), vocab {} | body {:?} embed {:?} | {} MB\n",
        model.name_str(),
        c.n_layers,
        c.dim,
        c.n_heads,
        c.n_kv_heads,
        c.head_dim,
        c.vocab_size,
        c.quant_body,
        c.quant_embed,
        model.file_bytes >> 20
    );
    if args.info {
        err!(
            "max_seq_len {}, rope_theta {}, rope {:?}, tied {}, bias {}, qk_norm {}, merges {}, pretok {}, chat {}, bos {}, stops {:?}\n",
            c.max_seq_len, c.rope_theta, c.rope, c.tied, c.qkv_bias, c.qk_norm, model.tok.n_merges, model.tok.pretok,
            model.tok.chat_kind, model.tok.bos, model.tok.stops
        );
        return 0;
    }
    let tok = match Tokenizer::new(&model) {
        Ok(t) => t,
        Err(e) => {
            err!("tokenizer: {}\n", e);
            return 1;
        }
    };

    if let Some(t) = args.encode {
        let mut v = Vec::new();
        tok.encode(t, &mut v);
        print_ids(&v);
        return 0;
    }
    if let Some(t) = args.chat_ids {
        let mut v = Vec::new();
        if !chat::user_turn(&tok, &model.tok, args.system, t, true, &mut v) {
            err!("model has no chat template\n");
            return 2;
        }
        print_ids(&v);
        return 0;
    }
    if args.encode_stdin {
        let data = sys::read_stdin_all();
        for rec in data.split(|&b| b == 0) {
            let mut v = Vec::new();
            match core::str::from_utf8(rec) {
                Ok(s) => tok.encode(s, &mut v),
                Err(_) => err!("skipping invalid UTF-8 record\n"),
            }
            print_ids(&v);
        }
        return 0;
    }
    if let Some(s) = args.decode {
        match parse_ids(s) {
            Some(ids) => {
                for id in ids {
                    sys::stdout(tok.token_bytes(id));
                }
                return 0;
            }
            None => {
                err!("bad id list\n");
                return 2;
            }
        }
    }

    let ctx = args
        .ctx
        .unwrap_or(c.max_seq_len.min(2048))
        .min(c.max_seq_len)
        .max(1);
    err!(
        "context {} tokens, KV cache {} MB\n",
        ctx,
        State::kv_bytes(c, ctx) >> 20
    );
    let mut st = State::new(c, ctx);
    let seed = args.seed.unwrap_or_else(|| sys::now_ms().max(1));
    let mut sampler = Sampler::new(args.temp, args.topp, seed);
    let mut pos = 0usize;
    let mut dump: Option<Vec<f32>> = args.dump.map(|_| Vec::new());
    let t0 = sys::now_ms();
    let mut generated = 0usize;
    let mut fed = 0usize;

    if args.chat || args.ask.is_some() {
        let mut ids = Vec::new();
        let mut turn = 0usize;
        let mut line = Vec::new();
        let mut one_shot = args.ask;
        loop {
            let msg: &str = if let Some(q) = one_shot.take() {
                q
            } else if args.ask.is_some() {
                break;
            } else {
                out!("> ");
                if !sys::read_line(&mut line) {
                    break;
                }
                match core::str::from_utf8(&line) {
                    Ok(s) => s,
                    Err(_) => {
                        err!("(input is not valid UTF-8)\n");
                        continue;
                    }
                }
            };
            let msg = msg.trim_end_matches('\r');
            if msg == "/quit" {
                break;
            }
            if msg.is_empty() {
                continue;
            }
            ids.clear();
            if !chat::user_turn(&tok, &model.tok, args.system, msg, turn == 0, &mut ids) {
                err!("this model has no chat template, use -p for raw completion\n");
                return 2;
            }
            if pos + ids.len() + 1 >= ctx {
                err!("[context full: {} + {} tokens > {}]\n", pos, ids.len(), ctx);
                break;
            }
            fed += ids.len();
            feed(&model, &mut st, &ids, &mut pos, &mut dump);
            generated += generate(&model, &tok, &mut st, &mut sampler, &mut pos, args.n);
            out!("\n");
            turn += 1;
        }
    } else {
        let mut ids = Vec::new();
        if let Some(s) = args.ids {
            match parse_ids(s) {
                Some(v) => ids = v,
                None => {
                    err!("bad id list\n");
                    return 2;
                }
            }
        } else if let Some(p) = args.prompt {
            if model.tok.bos >= 0 {
                ids.push(model.tok.bos as u32);
            }
            tok.encode(p, &mut ids);
            out!("{}", p);
        } else {
            err!("nothing to do: give -p, --ask, --chat, --ids, --encode, --decode or --info\n");
            return 2;
        }
        if ids.is_empty() {
            err!("empty prompt\n");
            return 2;
        }
        if let Some(bad) = ids.iter().find(|&&t| t as usize >= c.vocab_size) {
            err!(
                "token id {} is outside the vocabulary (size {})\n",
                bad,
                c.vocab_size
            );
            return 2;
        }
        if ids.len() + 1 >= ctx {
            err!("prompt has {} tokens, context is {}\n", ids.len(), ctx);
            return 2;
        }
        fed += ids.len();
        feed(&model, &mut st, &ids, &mut pos, &mut dump);
        if args.n > 0 {
            generated += generate(&model, &tok, &mut st, &mut sampler, &mut pos, args.n);
            out!("\n");
        }
    }

    if let (Some(path), Some(d)) = (args.dump, dump.as_ref()) {
        let bytes = unsafe { core::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
        if let Err(e) = sys::write_file(path, bytes) {
            err!("{}: {}\n", text(path), e);
            return 1;
        }
    }
    let ms = sys::now_ms().saturating_sub(t0);
    let total = fed + generated;
    if ms > 0 {
        err!(
            "[{} prompt + {} generated tokens in {} ms = {:.2} tok/s, seed {}]\n",
            fed,
            generated,
            ms,
            total as f32 * 1000.0 / ms as f32,
            seed
        );
    }
    0
}

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const c_char) -> i32 {
    let mut args: Vec<&'static [u8]> = Vec::new();
    for i in 0..argc.max(0) as usize {
        args.push(cstr(unsafe { *argv.add(i) }));
    }
    run(&args)
}
