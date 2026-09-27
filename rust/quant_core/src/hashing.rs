//! SHA-256 hex digests. Output matches `hashlib.sha256(...).hexdigest()`.

use sha2::{Digest, Sha256};

const HEX: [u8; 16] = *b"0123456789abcdef";

/// Lowercase hex of the SHA-256 digest, on the stack. Always 64 ASCII bytes.
pub fn sha256_hex_ascii(data: &[u8]) -> [u8; 64] {
    let digest = Sha256::digest(data);
    let mut out = [0u8; 64];
    for (i, byte) in digest.iter().enumerate() {
        out[i * 2] = HEX[(*byte >> 4) as usize];
        out[i * 2 + 1] = HEX[(*byte & 0x0f) as usize];
    }
    out
}

pub fn sha256_hex(data: &[u8]) -> String {
    let bytes = sha256_hex_ascii(data);
    match std::str::from_utf8(&bytes) {
        Ok(text) => text.to_owned(),
        Err(_) => unreachable!("hex digits are ASCII"),
    }
}
