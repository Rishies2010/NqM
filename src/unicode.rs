use crate::unicode_tables::{LETTER, NUMBER};

fn in_table(t: &[(u32, u32)], c: u32) -> bool {
    t.binary_search_by(|&(lo, hi)| {
        if c < lo {
            core::cmp::Ordering::Greater
        } else if c > hi {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

#[inline]
pub fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_table(&LETTER, c as u32)
}

#[inline]
pub fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_table(&NUMBER, c as u32)
}

#[inline]
pub fn is_space(c: char) -> bool {
    c.is_whitespace()
}

#[inline]
pub fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}
