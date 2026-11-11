#![cfg_attr(not(test), no_std)]
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

extern crate alloc;

#[cfg(target_endian = "big")]
compile_error!("NQM1 is little-endian only");

pub mod buf;
pub mod chat;
pub mod error;
pub mod format;
pub mod math;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod sys;
pub mod tokenizer;
pub mod unicode;
mod unicode_tables;

pub use error::{Error, Result};
