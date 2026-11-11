use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Io(&'static str),
    BadMagic,
    BadVersion(u32),
    Truncated,
    Misaligned,
    Unsupported(&'static str),
    Invalid(&'static str),
}

pub type Result<T> = core::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(s) => write!(f, "io error: {s}"),
            Error::BadMagic => write!(f, "not an NQM1 file (bad magic)"),
            Error::BadVersion(v) => write!(f, "unsupported NQM version {v}"),
            Error::Truncated => write!(f, "file is truncated or sizes do not add up"),
            Error::Misaligned => write!(f, "buffer must be 64-byte aligned"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Invalid(s) => write!(f, "invalid file: {s}"),
        }
    }
}
