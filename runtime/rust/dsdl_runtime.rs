#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

#[cfg(feature = "std")]
pub type DsdlVec<T> = std::vec::Vec<T>;

#[cfg(not(feature = "std"))]
pub type DsdlVec<T> = alloc::vec::Vec<T>;

pub const DSDL_RUNTIME_SUCCESS: i8 = 0;
pub const DSDL_RUNTIME_ERROR_INVALID_ARGUMENT: i8 = 2;
pub const DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL: i8 = 3;
pub const DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH: i8 = 10;
pub const DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG: i8 = 11;
pub const DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER: i8 = 12;

#[inline]
pub fn choose_min(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn saturate_fragment_bits(
    buffer_size_bytes: usize,
    fragment_offset_bits: usize,
    fragment_length_bits: usize,
) -> usize {
    let size_bits = buffer_size_bytes.saturating_mul(8);
    let tail_bits = size_bits.saturating_sub(choose_min(size_bits, fragment_offset_bits));
    choose_min(fragment_length_bits, tail_bits)
}

#[inline]
fn copy_bits_portable(
    dst: &mut [u8],
    dst_offset_bits: usize,
    length_bits: usize,
    src: &[u8],
    src_offset_bits: usize,
) {
    for i in 0..length_bits {
        let src_bit_index = src_offset_bits + i;
        if (src_bit_index / 8) >= src.len() {
            break;
        }

        let dst_bit_index = dst_offset_bits + i;
        if (dst_bit_index / 8) >= dst.len() {
            break;
        }

        let bit = ((src[src_bit_index / 8] >> (src_bit_index % 8)) & 1u8) != 0;
        if bit {
            dst[dst_bit_index / 8] |= 1u8 << (dst_bit_index % 8);
        } else {
            dst[dst_bit_index / 8] &= !(1u8 << (dst_bit_index % 8));
        }
    }
}

#[inline]
#[cfg(feature = "runtime-fast")]
fn copy_bits_runtime_fast(
    dst: &mut [u8],
    dst_offset_bits: usize,
    length_bits: usize,
    src: &[u8],
    src_offset_bits: usize,
) {
    if length_bits == 0 {
        return;
    }
    if (dst_offset_bits | src_offset_bits | length_bits) & 7 == 0 {
        let dst_start = dst_offset_bits / 8;
        let src_start = src_offset_bits / 8;
        if dst_start <= dst.len() && src_start <= src.len() {
            let dst_available = dst.len().saturating_sub(dst_start);
            let src_available = src.len().saturating_sub(src_start);
            let requested_bytes = length_bits / 8;
            let copy_bytes = choose_min(requested_bytes, choose_min(dst_available, src_available));
            if copy_bytes > 0 {
                dst[dst_start..dst_start + copy_bytes]
                    .copy_from_slice(&src[src_start..src_start + copy_bytes]);
            }
            return;
        }
    }
    copy_bits_portable(dst, dst_offset_bits, length_bits, src, src_offset_bits);
}

#[inline]
pub fn copy_bits(
    dst: &mut [u8],
    dst_offset_bits: usize,
    length_bits: usize,
    src: &[u8],
    src_offset_bits: usize,
) {
    #[cfg(feature = "runtime-fast")]
    {
        copy_bits_runtime_fast(dst, dst_offset_bits, length_bits, src, src_offset_bits);
        return;
    }

    #[cfg(not(feature = "runtime-fast"))]
    {
        copy_bits_portable(dst, dst_offset_bits, length_bits, src, src_offset_bits);
    }
}

#[inline]
pub fn get_bits(output: &mut [u8], buf: &[u8], off_bits: usize, len_bits: usize) {
    let sat_bits = saturate_fragment_bits(buf.len(), off_bits, len_bits);
    let out_len = (len_bits + 7) / 8;
    for b in output.iter_mut().take(out_len) {
        *b = 0;
    }
    copy_bits(output, 0, sat_bits, buf, off_bits);
}

#[inline]
pub fn set_bit(buf: &mut [u8], off_bits: usize, value: bool) -> i8 {
    if buf.len().saturating_mul(8) <= off_bits {
        return -DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL;
    }
    let val = if value { 1u8 } else { 0u8 };
    copy_bits(buf, off_bits, 1, &[val], 0);
    DSDL_RUNTIME_SUCCESS
}

#[inline]
pub fn set_uxx(buf: &mut [u8], off_bits: usize, value: u64, len_bits: u8) -> i8 {
    if buf.len().saturating_mul(8) < off_bits.saturating_add(len_bits as usize) {
        return -DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL;
    }
    let saturated_len_bits = choose_min(len_bits as usize, 64);
    let tmp = value.to_le_bytes();
    copy_bits(buf, off_bits, saturated_len_bits, &tmp, 0);
    DSDL_RUNTIME_SUCCESS
}

#[inline]
pub fn set_ixx(buf: &mut [u8], off_bits: usize, value: i64, len_bits: u8) -> i8 {
    set_uxx(buf, off_bits, value as u64, len_bits)
}

#[inline]
pub fn get_bit(buf: &[u8], off_bits: usize) -> bool {
    get_u8(buf, off_bits, 1) == 1
}

#[inline]
pub fn get_u8(buf: &[u8], off_bits: usize, len_bits: u8) -> u8 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 8));
    let mut out = [0u8; 1];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    out[0]
}

#[inline]
pub fn get_u16(buf: &[u8], off_bits: usize, len_bits: u8) -> u16 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 16));
    let mut out = [0u8; 2];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    u16::from_le_bytes(out)
}

#[inline]
pub fn get_u32(buf: &[u8], off_bits: usize, len_bits: u8) -> u32 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 32));
    let mut out = [0u8; 4];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    u32::from_le_bytes(out)
}

#[inline]
pub fn get_u64(buf: &[u8], off_bits: usize, len_bits: u8) -> u64 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 64));
    let mut out = [0u8; 8];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    u64::from_le_bytes(out)
}

#[inline]
fn sign_extend_u64(value: u64, sat: u8) -> u64 {
    if sat == 0 || sat >= 64 {
        return value;
    }
    let sign_bit = 1u64 << (sat - 1);
    if (value & sign_bit) == 0 {
        return value;
    }
    value | (!0u64 << sat)
}

#[inline]
pub fn get_i8(buf: &[u8], off_bits: usize, len_bits: u8) -> i8 {
    let sat = choose_min(len_bits as usize, 8) as u8;
    let val = get_u8(buf, off_bits, sat) as u64;
    sign_extend_u64(val, sat) as i8
}

#[inline]
pub fn get_i16(buf: &[u8], off_bits: usize, len_bits: u8) -> i16 {
    let sat = choose_min(len_bits as usize, 16) as u8;
    let val = get_u16(buf, off_bits, sat) as u64;
    sign_extend_u64(val, sat) as i16
}

#[inline]
pub fn get_i32(buf: &[u8], off_bits: usize, len_bits: u8) -> i32 {
    let sat = choose_min(len_bits as usize, 32) as u8;
    let val = get_u32(buf, off_bits, sat) as u64;
    sign_extend_u64(val, sat) as i32
}

#[inline]
pub fn get_i64(buf: &[u8], off_bits: usize, len_bits: u8) -> i64 {
    let sat = choose_min(len_bits as usize, 64) as u8;
    let val = get_u64(buf, off_bits, sat);
    sign_extend_u64(val, sat) as i64
}

pub fn float16_pack(value: f32) -> u16 {
    let round_mask: u32 = !0x0FFF;
    let f32inf: u32 = 255u32 << 23;
    let f16inf: u32 = 31u32 << 23;
    let magic: u32 = 15u32 << 23;

    let mut in_bits = value.to_bits();
    let sign = in_bits & (1u32 << 31);
    in_bits ^= sign;
    let out: u16;

    if in_bits >= f32inf {
        out = if (in_bits & 0x7F_FFFF) != 0 {
            0x7E00
        } else if in_bits > f32inf {
            0x7FFF
        } else {
            0x7C00
        };
    } else {
        in_bits &= round_mask;
        let scaled = f32::from_bits(in_bits) * f32::from_bits(magic);
        in_bits = scaled.to_bits();
        in_bits = in_bits.saturating_sub(round_mask);
        if in_bits > f16inf {
            in_bits = f16inf;
        }
        out = (in_bits >> 13) as u16;
    }

    out | ((sign >> 16) as u16)
}

pub fn float16_unpack(value: u16) -> f32 {
    let magic = 0xEFu32 << 23;
    let inf_nan = 0x8Fu32 << 23;
    let mut out_bits = ((value & 0x7FFF) as u32) << 13;
    let scaled = f32::from_bits(out_bits) * f32::from_bits(magic);
    out_bits = scaled.to_bits();
    if scaled >= f32::from_bits(inf_nan) {
        out_bits |= 0xFFu32 << 23;
    }
    out_bits |= ((value & 0x8000) as u32) << 16;
    f32::from_bits(out_bits)
}

#[inline]
pub fn set_f16(buf: &mut [u8], off_bits: usize, value: f32) -> i8 {
    set_uxx(buf, off_bits, float16_pack(value) as u64, 16)
}

#[inline]
pub fn get_f16(buf: &[u8], off_bits: usize) -> f32 {
    float16_unpack(get_u16(buf, off_bits, 16))
}

#[inline]
pub fn set_f32(buf: &mut [u8], off_bits: usize, value: f32) -> i8 {
    set_uxx(buf, off_bits, value.to_bits() as u64, 32)
}

#[inline]
pub fn get_f32(buf: &[u8], off_bits: usize) -> f32 {
    f32::from_bits(get_u32(buf, off_bits, 32))
}

#[inline]
pub fn set_f64(buf: &mut [u8], off_bits: usize, value: f64) -> i8 {
    set_uxx(buf, off_bits, value.to_bits(), 64)
}

#[inline]
pub fn get_f64(buf: &[u8], off_bits: usize) -> f64 {
    f64::from_bits(get_u64(buf, off_bits, 64))
}
