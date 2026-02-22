#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

//! Portable bit-level runtime helpers for generated DSDL Rust bindings.
//!
//! This module provides integer/float serialization primitives and bounded
//! bit-copy helpers shared by generated serializers and deserializers.

use core::fmt;
use core::ops::{Deref, DerefMut, Index, IndexMut};

#[cfg(feature = "std")]
type DsdlBackingVec<T> = std::vec::Vec<T>;

#[cfg(not(feature = "std"))]
type DsdlBackingVec<T> = alloc::vec::Vec<T>;

/// Runtime memory strategy selected for variable-length fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsdlMemoryMode {
    /// Uses only inline capacity semantics (deterministic, no pool path).
    MaxInline,
    /// Uses inline capacity below threshold and pool allocation above threshold.
    InlineThenPool,
}

/// Stable class identifier for per-type/per-field allocation groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AllocationClassId(pub u32);

/// Memory contract associated with a variable-length runtime container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarArrayMemoryContract {
    /// Configured memory mode.
    pub mode: DsdlMemoryMode,
    /// Inline threshold in bytes for [`DsdlMemoryMode::InlineThenPool`].
    pub inline_threshold_bytes: usize,
    /// Allocation class for pool requests.
    pub allocation_class: AllocationClassId,
}

impl VarArrayMemoryContract {
    /// Creates a memory contract.
    pub const fn new(
        mode: DsdlMemoryMode,
        inline_threshold_bytes: usize,
        allocation_class: AllocationClassId,
    ) -> Self {
        Self {
            mode,
            inline_threshold_bytes,
            allocation_class,
        }
    }
}

impl Default for VarArrayMemoryContract {
    fn default() -> Self {
        Self {
            mode: DsdlMemoryMode::MaxInline,
            inline_threshold_bytes: usize::MAX,
            allocation_class: AllocationClassId(0),
        }
    }
}

/// Allocation failure categories used by pool-backed memory mode handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationErrorKind {
    /// Requested allocation exceeds pool capacity.
    OutOfMemory,
    /// Allocation parameters are invalid (for example bad alignment).
    InvalidRequest,
    /// No pool implementation is available for the requested mode.
    PoolUnavailable,
}

impl AllocationErrorKind {
    /// Returns the stable textual identifier for this allocation error kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutOfMemory => "out_of_memory",
            Self::InvalidRequest => "invalid_request",
            Self::PoolUnavailable => "pool_unavailable",
        }
    }
}

impl fmt::Display for AllocationErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stable allocation failure payload for generated/runtime diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationError {
    /// Failure category.
    pub kind: AllocationErrorKind,
    /// Associated allocation class.
    pub class_id: AllocationClassId,
    /// Requested allocation size in bytes.
    pub requested_bytes: usize,
}

impl AllocationError {
    fn new(kind: AllocationErrorKind, class_id: AllocationClassId, requested_bytes: usize) -> Self {
        Self {
            kind,
            class_id,
            requested_bytes,
        }
    }
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "allocation_error(kind={}, class_id={}, requested_bytes={})",
            self.kind.as_str(),
            self.class_id.0,
            self.requested_bytes
        )
    }
}

/// Runtime success code.
pub const DSDL_RUNTIME_SUCCESS: i8 = 0;
/// API usage error code for invalid arguments.
pub const DSDL_RUNTIME_ERROR_INVALID_ARGUMENT: i8 = 2;
/// API usage error code for insufficient serialization buffer size.
pub const DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL: i8 = 3;
/// Representation error code for malformed array-length values.
pub const DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH: i8 = 10;
/// Representation error code for malformed union-tag values.
pub const DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG: i8 = 11;
/// Representation error code for malformed delimiter headers.
pub const DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER: i8 = 12;
/// Runtime error code for pool allocation failures.
pub const DSDL_RUNTIME_ERROR_ALLOCATION_OUT_OF_MEMORY: i8 = 13;
/// Runtime error code for missing/disabled pool provider.
pub const DSDL_RUNTIME_ERROR_ALLOCATION_POOL_UNAVAILABLE: i8 = 14;
/// Runtime error code for invalid pool allocation request parameters.
pub const DSDL_RUNTIME_ERROR_ALLOCATION_INVALID_REQUEST: i8 = 15;

/// Maps a stable allocation error into a runtime error code.
pub fn allocation_error_to_runtime_code(error: AllocationError) -> i8 {
    match error.kind {
        AllocationErrorKind::OutOfMemory => DSDL_RUNTIME_ERROR_ALLOCATION_OUT_OF_MEMORY,
        AllocationErrorKind::InvalidRequest => DSDL_RUNTIME_ERROR_ALLOCATION_INVALID_REQUEST,
        AllocationErrorKind::PoolUnavailable => DSDL_RUNTIME_ERROR_ALLOCATION_POOL_UNAVAILABLE,
    }
}

/// Returns troubleshooting guidance for allocation failure categories.
pub const fn allocation_error_hint(kind: AllocationErrorKind) -> &'static str {
    match kind {
        AllocationErrorKind::OutOfMemory => {
            "Increase per-type pool budget or use max-inline mode for deterministic pre-allocation."
        }
        AllocationErrorKind::InvalidRequest => {
            "Verify allocation request size/alignment and generated pool contract metadata."
        }
        AllocationErrorKind::PoolUnavailable => {
            "Enable/configure a pool provider for inline-then-pool mode, or switch to max-inline mode."
        }
    }
}

/// Allocator interface used by pool-backed variable-length storage strategies.
pub trait PoolProvider {
    /// Provider-specific allocation handle.
    type Handle: Copy;

    /// Allocates `bytes` at `align` for `class_id`.
    fn alloc(
        &mut self,
        class_id: AllocationClassId,
        bytes: usize,
        align: usize,
    ) -> Result<Self::Handle, AllocationError>;

    /// Releases a previously-issued handle.
    fn dealloc(&mut self, class_id: AllocationClassId, handle: Self::Handle);
}

/// Handle type for [`NullPoolProvider`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NullPoolHandle;

/// Reference pool provider that always reports pool unavailability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NullPoolProvider;

impl PoolProvider for NullPoolProvider {
    type Handle = NullPoolHandle;

    fn alloc(
        &mut self,
        class_id: AllocationClassId,
        bytes: usize,
        _align: usize,
    ) -> Result<Self::Handle, AllocationError> {
        Err(AllocationError::new(
            AllocationErrorKind::PoolUnavailable,
            class_id,
            bytes,
        ))
    }

    fn dealloc(&mut self, _class_id: AllocationClassId, _handle: Self::Handle) {}
}

/// Handle type for [`PassthroughPoolProvider`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassthroughPoolHandle(pub u64);

/// Reference pool provider that accepts all allocation requests.
///
/// This provider is useful for preserving existing generated behavior while
/// still exercising pool allocation routes in code paths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassthroughPoolProvider {
    next_handle: u64,
}

impl PoolProvider for PassthroughPoolProvider {
    type Handle = PassthroughPoolHandle;

    fn alloc(
        &mut self,
        class_id: AllocationClassId,
        bytes: usize,
        align: usize,
    ) -> Result<Self::Handle, AllocationError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(AllocationError::new(
                AllocationErrorKind::InvalidRequest,
                class_id,
                bytes,
            ));
        }
        let handle = PassthroughPoolHandle(self.next_handle);
        self.next_handle = self.next_handle.wrapping_add(1);
        Ok(handle)
    }

    fn dealloc(&mut self, _class_id: AllocationClassId, _handle: Self::Handle) {}
}

/// Handle type for [`BudgetPoolProvider`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetPoolHandle(pub u64);

/// Reference pool provider with a fixed byte budget.
///
/// This provider enforces deterministic boundary behavior for tests and
/// embedded bring-up. Deallocation is intentionally a no-op in this reference
/// implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetPoolProvider {
    remaining_bytes: usize,
    total_allocated_bytes: usize,
    allocations: u64,
    next_handle: u64,
}

impl BudgetPoolProvider {
    /// Constructs a provider with the specified total budget.
    pub fn new(total_budget_bytes: usize) -> Self {
        Self {
            remaining_bytes: total_budget_bytes,
            total_allocated_bytes: 0,
            allocations: 0,
            next_handle: 1,
        }
    }

    /// Returns the remaining byte budget.
    pub fn remaining_bytes(&self) -> usize {
        self.remaining_bytes
    }

    /// Returns the total number of successful allocation calls.
    pub fn allocations(&self) -> u64 {
        self.allocations
    }

    /// Returns the total bytes consumed by successful allocations.
    pub fn total_allocated_bytes(&self) -> usize {
        self.total_allocated_bytes
    }
}

fn align_up(value: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

impl PoolProvider for BudgetPoolProvider {
    type Handle = BudgetPoolHandle;

    fn alloc(
        &mut self,
        class_id: AllocationClassId,
        bytes: usize,
        align: usize,
    ) -> Result<Self::Handle, AllocationError> {
        let aligned_bytes = match align_up(bytes, align) {
            Some(v) => v,
            None => {
                return Err(AllocationError::new(
                    AllocationErrorKind::InvalidRequest,
                    class_id,
                    bytes,
                ));
            }
        };
        if aligned_bytes > self.remaining_bytes {
            return Err(AllocationError::new(
                AllocationErrorKind::OutOfMemory,
                class_id,
                aligned_bytes,
            ));
        }
        self.remaining_bytes -= aligned_bytes;
        self.total_allocated_bytes += aligned_bytes;
        self.allocations += 1;
        let handle = BudgetPoolHandle(self.next_handle);
        self.next_handle = self.next_handle.wrapping_add(1);
        Ok(handle)
    }

    fn dealloc(&mut self, _class_id: AllocationClassId, _handle: Self::Handle) {}
}

/// Variable-length storage wrapper used by generated Rust bindings.
///
/// This wrapper preserves the historical `Vec`-like API used by generated
/// code, while adding memory-mode contract metadata and optional pool-aware
/// capacity reservation hooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarArray<T> {
    inner: DsdlBackingVec<T>,
    contract: VarArrayMemoryContract,
}

impl<T> VarArray<T> {
    /// Constructs an empty array with default memory contract.
    pub fn new() -> Self {
        Self {
            inner: DsdlBackingVec::new(),
            contract: VarArrayMemoryContract::default(),
        }
    }

    /// Constructs an empty array with default contract and capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: DsdlBackingVec::with_capacity(capacity),
            contract: VarArrayMemoryContract::default(),
        }
    }

    /// Constructs an empty array using an explicit memory contract.
    pub fn with_contract(contract: VarArrayMemoryContract) -> Self {
        Self {
            inner: DsdlBackingVec::new(),
            contract,
        }
    }

    /// Constructs an array with explicit memory contract and capacity.
    pub fn with_contract_and_capacity(contract: VarArrayMemoryContract, capacity: usize) -> Self {
        Self {
            inner: DsdlBackingVec::with_capacity(capacity),
            contract,
        }
    }

    /// Returns the current memory contract.
    pub fn memory_contract(&self) -> VarArrayMemoryContract {
        self.contract
    }

    /// Replaces the memory contract.
    pub fn set_memory_contract(&mut self, contract: VarArrayMemoryContract) {
        self.contract = contract;
    }

    /// Returns number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true when empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns current backing capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Removes all elements.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Appends one element.
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    /// Ensures capacity for at least `additional` extra elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Ensures capacity while honoring pool-mode allocation contract.
    ///
    /// In `max-inline` mode this behaves like [`VarArray::reserve`]. In
    /// `inline-then-pool` mode, crossing the inline threshold triggers a pool
    /// allocation request before reserving backing storage.
    pub fn reserve_with_pool<P: PoolProvider>(
        &mut self,
        additional: usize,
        pool: &mut P,
    ) -> Result<(), AllocationError> {
        if self.contract.mode == DsdlMemoryMode::InlineThenPool {
            let elem_bytes = core::mem::size_of::<T>();
            if elem_bytes != 0 {
                let target_len = self.inner.len().saturating_add(additional);
                let target_bytes = target_len.saturating_mul(elem_bytes);
                if target_bytes > self.contract.inline_threshold_bytes {
                    let requested = target_bytes - self.contract.inline_threshold_bytes;
                    let align = core::mem::align_of::<T>().max(1);
                    let _ = pool.alloc(self.contract.allocation_class, requested, align)?;
                }
            }
        }
        self.inner.reserve(additional);
        Ok(())
    }

    /// Resizes the array to `new_len`, cloning `value` when needed.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.inner.resize(new_len, value);
    }

    /// Truncates the array to `len` elements.
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Returns immutable slice view.
    pub fn as_slice(&self) -> &[T] {
        &self.inner
    }

    /// Returns mutable slice view.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.inner
    }

    /// Consumes the wrapper and returns the backing vector.
    pub fn into_inner(self) -> DsdlBackingVec<T> {
        self.inner
    }
}

impl<T> Default for VarArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for VarArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for VarArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T> Index<usize> for VarArray<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<T> IndexMut<usize> for VarArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<T> AsRef<[T]> for VarArray<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for VarArray<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Vector wrapper used by generated code.
pub type DsdlVec<T> = VarArray<T>;

#[inline]
/// Returns the smaller of two values.
pub fn choose_min(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
/// Saturates a requested bit fragment to fit inside a bounded byte buffer.
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
/// Portable bit-copy implementation used when fast-path conditions do not hold.
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
/// Optimized bit-copy implementation with aligned-copy fast path.
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
/// Copies `length_bits` from `src` to `dst` using DSDL bit ordering.
///
/// When `runtime-fast` is enabled, aligned copies are accelerated.
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
/// Reads a bit fragment from `buf` into `output` with implicit zero extension.
pub fn get_bits(output: &mut [u8], buf: &[u8], off_bits: usize, len_bits: usize) {
    let sat_bits = saturate_fragment_bits(buf.len(), off_bits, len_bits);
    let out_len = (len_bits + 7) / 8;
    for b in output.iter_mut().take(out_len) {
        *b = 0;
    }
    copy_bits(output, 0, sat_bits, buf, off_bits);
}

#[inline]
/// Serializes one boolean bit into `buf` at `off_bits`.
///
/// Returns [`DSDL_RUNTIME_SUCCESS`] on success or a negative error code.
pub fn set_bit(buf: &mut [u8], off_bits: usize, value: bool) -> i8 {
    if buf.len().saturating_mul(8) <= off_bits {
        return -DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL;
    }
    let val = if value { 1u8 } else { 0u8 };
    copy_bits(buf, off_bits, 1, &[val], 0);
    DSDL_RUNTIME_SUCCESS
}

#[inline]
/// Serializes an unsigned integer fragment of `len_bits` width.
///
/// Returns [`DSDL_RUNTIME_SUCCESS`] on success or a negative error code.
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
/// Serializes a signed integer fragment of `len_bits` width.
pub fn set_ixx(buf: &mut [u8], off_bits: usize, value: i64, len_bits: u8) -> i8 {
    set_uxx(buf, off_bits, value as u64, len_bits)
}

#[inline]
/// Deserializes one bit as boolean.
pub fn get_bit(buf: &[u8], off_bits: usize) -> bool {
    get_u8(buf, off_bits, 1) == 1
}

#[inline]
/// Deserializes up to 8 bits as an unsigned integer.
pub fn get_u8(buf: &[u8], off_bits: usize, len_bits: u8) -> u8 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 8));
    let mut out = [0u8; 1];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    out[0]
}

#[inline]
/// Deserializes up to 16 bits as an unsigned integer.
pub fn get_u16(buf: &[u8], off_bits: usize, len_bits: u8) -> u16 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 16));
    let mut out = [0u8; 2];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    u16::from_le_bytes(out)
}

#[inline]
/// Deserializes up to 32 bits as an unsigned integer.
pub fn get_u32(buf: &[u8], off_bits: usize, len_bits: u8) -> u32 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 32));
    let mut out = [0u8; 4];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    u32::from_le_bytes(out)
}

#[inline]
/// Deserializes up to 64 bits as an unsigned integer.
pub fn get_u64(buf: &[u8], off_bits: usize, len_bits: u8) -> u64 {
    let bits = saturate_fragment_bits(buf.len(), off_bits, choose_min(len_bits as usize, 64));
    let mut out = [0u8; 8];
    copy_bits(&mut out, 0, bits, buf, off_bits);
    u64::from_le_bytes(out)
}

#[inline]
/// Sign-extends an unsigned integer interpreted with `sat` significant bits.
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
/// Deserializes up to 8 bits as a signed integer.
pub fn get_i8(buf: &[u8], off_bits: usize, len_bits: u8) -> i8 {
    let sat = choose_min(len_bits as usize, 8) as u8;
    let val = get_u8(buf, off_bits, sat) as u64;
    sign_extend_u64(val, sat) as i8
}

#[inline]
/// Deserializes up to 16 bits as a signed integer.
pub fn get_i16(buf: &[u8], off_bits: usize, len_bits: u8) -> i16 {
    let sat = choose_min(len_bits as usize, 16) as u8;
    let val = get_u16(buf, off_bits, sat) as u64;
    sign_extend_u64(val, sat) as i16
}

#[inline]
/// Deserializes up to 32 bits as a signed integer.
pub fn get_i32(buf: &[u8], off_bits: usize, len_bits: u8) -> i32 {
    let sat = choose_min(len_bits as usize, 32) as u8;
    let val = get_u32(buf, off_bits, sat) as u64;
    sign_extend_u64(val, sat) as i32
}

#[inline]
/// Deserializes up to 64 bits as a signed integer.
pub fn get_i64(buf: &[u8], off_bits: usize, len_bits: u8) -> i64 {
    let sat = choose_min(len_bits as usize, 64) as u8;
    let val = get_u64(buf, off_bits, sat);
    sign_extend_u64(val, sat) as i64
}

/// Converts an IEEE-754 `f32` value into binary16 encoding.
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

/// Converts a binary16 value into IEEE-754 `f32`.
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
/// Serializes `value` as binary16 at `off_bits`.
pub fn set_f16(buf: &mut [u8], off_bits: usize, value: f32) -> i8 {
    set_uxx(buf, off_bits, float16_pack(value) as u64, 16)
}

#[inline]
/// Deserializes a binary16 value at `off_bits`.
pub fn get_f16(buf: &[u8], off_bits: usize) -> f32 {
    float16_unpack(get_u16(buf, off_bits, 16))
}

#[inline]
/// Serializes `value` as IEEE-754 binary32 at `off_bits`.
pub fn set_f32(buf: &mut [u8], off_bits: usize, value: f32) -> i8 {
    set_uxx(buf, off_bits, value.to_bits() as u64, 32)
}

#[inline]
/// Deserializes an IEEE-754 binary32 value at `off_bits`.
pub fn get_f32(buf: &[u8], off_bits: usize) -> f32 {
    f32::from_bits(get_u32(buf, off_bits, 32))
}

#[inline]
/// Serializes `value` as IEEE-754 binary64 at `off_bits`.
pub fn set_f64(buf: &mut [u8], off_bits: usize, value: f64) -> i8 {
    set_uxx(buf, off_bits, value.to_bits(), 64)
}

#[inline]
/// Deserializes an IEEE-754 binary64 value at `off_bits`.
pub fn get_f64(buf: &[u8], off_bits: usize) -> f64 {
    f64::from_bits(get_u64(buf, off_bits, 64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_array_vec_compatible_surface() {
        let mut values = DsdlVec::<u8>::with_capacity(2);
        assert_eq!(values.len(), 0);
        assert!(values.is_empty());
        values.push(3);
        values.push(5);
        values.resize(4, 9);
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], 3);
        assert_eq!(values[1], 5);
        assert_eq!(values[2], 9);
        assert_eq!(values[3], 9);
        values.clear();
        assert!(values.is_empty());
    }

    #[test]
    fn reserve_with_pool_max_inline_does_not_allocate() {
        let mut values = DsdlVec::<u8>::with_contract(VarArrayMemoryContract::new(
            DsdlMemoryMode::MaxInline,
            8,
            AllocationClassId(7),
        ));
        let mut pool = BudgetPoolProvider::new(16);
        values
            .reserve_with_pool(64, &mut pool)
            .expect("max-inline reserve_with_pool should not hit pool allocation");
        assert_eq!(pool.allocations(), 0);
        assert_eq!(pool.remaining_bytes(), 16);
    }

    #[test]
    fn reserve_with_pool_inline_then_pool_threshold_behavior() {
        let contract =
            VarArrayMemoryContract::new(DsdlMemoryMode::InlineThenPool, 16, AllocationClassId(42));
        let mut values = DsdlVec::<u8>::with_contract(contract);
        let mut pool = BudgetPoolProvider::new(32);

        values
            .reserve_with_pool(8, &mut pool)
            .expect("below-threshold reserve should not allocate");
        assert_eq!(pool.allocations(), 0);
        assert_eq!(pool.remaining_bytes(), 32);

        values
            .reserve_with_pool(24, &mut pool)
            .expect("above-threshold reserve should allocate");
        assert_eq!(pool.allocations(), 1);
        assert_eq!(pool.remaining_bytes(), 24);
        assert_eq!(pool.total_allocated_bytes(), 8);
    }

    #[test]
    fn reserve_with_pool_inline_then_pool_oom_reports_taxonomy() {
        let contract =
            VarArrayMemoryContract::new(DsdlMemoryMode::InlineThenPool, 8, AllocationClassId(11));
        let mut values = DsdlVec::<u16>::with_contract(contract);
        let mut pool = BudgetPoolProvider::new(4);

        let err = values
            .reserve_with_pool(8, &mut pool)
            .expect_err("reserve_with_pool should fail with out-of-memory");
        assert_eq!(err.kind, AllocationErrorKind::OutOfMemory);
        assert_eq!(err.class_id, AllocationClassId(11));
        assert_eq!(
            allocation_error_to_runtime_code(err),
            DSDL_RUNTIME_ERROR_ALLOCATION_OUT_OF_MEMORY
        );
    }

    #[test]
    fn null_pool_provider_reports_unavailable() {
        let mut null_pool = NullPoolProvider;
        let err = PoolProvider::alloc(&mut null_pool, AllocationClassId(3), 16, 8)
            .expect_err("null provider should always fail allocations");
        assert_eq!(err.kind, AllocationErrorKind::PoolUnavailable);
        assert_eq!(err.class_id, AllocationClassId(3));
        assert_eq!(err.requested_bytes, 16);
        assert_eq!(
            allocation_error_to_runtime_code(err),
            DSDL_RUNTIME_ERROR_ALLOCATION_POOL_UNAVAILABLE
        );
    }

    #[test]
    fn budget_pool_provider_rejects_invalid_alignment() {
        let mut pool = BudgetPoolProvider::new(32);
        let err = PoolProvider::alloc(&mut pool, AllocationClassId(9), 12, 3)
            .expect_err("non-power-of-two alignment should be rejected");
        assert_eq!(err.kind, AllocationErrorKind::InvalidRequest);
        assert_eq!(err.class_id, AllocationClassId(9));
        assert_eq!(err.requested_bytes, 12);
        assert_eq!(
            allocation_error_to_runtime_code(err),
            DSDL_RUNTIME_ERROR_ALLOCATION_INVALID_REQUEST
        );
    }

    #[test]
    fn reserve_with_pool_failure_keeps_existing_state() {
        let contract =
            VarArrayMemoryContract::new(DsdlMemoryMode::InlineThenPool, 0, AllocationClassId(77));
        let mut values = DsdlVec::<u8>::with_contract(contract);
        values.push(11);
        values.push(22);
        values.push(33);
        let capacity_before = values.capacity();

        let mut pool = BudgetPoolProvider::new(1);
        let err = values
            .reserve_with_pool(8, &mut pool)
            .expect_err("tiny budget should force deterministic OOM");
        assert_eq!(err.kind, AllocationErrorKind::OutOfMemory);
        assert_eq!(values.as_slice(), &[11, 22, 33]);
        assert_eq!(values.len(), 3);
        assert_eq!(values.capacity(), capacity_before);
        assert_eq!(pool.allocations(), 0);
    }

    #[test]
    fn reserve_with_pool_zst_never_allocates() {
        let contract =
            VarArrayMemoryContract::new(DsdlMemoryMode::InlineThenPool, 0, AllocationClassId(3));
        let mut values = DsdlVec::<()>::with_contract(contract);
        let mut pool = BudgetPoolProvider::new(0);
        values
            .reserve_with_pool(1024, &mut pool)
            .expect("zero-sized elements should not allocate from pool");
        assert_eq!(values.len(), 0);
        assert_eq!(pool.allocations(), 0);
        assert_eq!(pool.remaining_bytes(), 0);
    }

    #[test]
    fn tiny_budget_failures_are_deterministic_across_calls() {
        let contract =
            VarArrayMemoryContract::new(DsdlMemoryMode::InlineThenPool, 0, AllocationClassId(5));
        let mut values = DsdlVec::<u8>::with_contract(contract);
        let mut pool = BudgetPoolProvider::new(4);

        values
            .reserve_with_pool(4, &mut pool)
            .expect("first reserve should fit tiny budget");
        assert_eq!(pool.allocations(), 1);
        assert_eq!(pool.remaining_bytes(), 0);

        let err = values
            .reserve_with_pool(8, &mut pool)
            .expect_err("second reserve should deterministically fail after budget exhaustion");
        assert_eq!(err.kind, AllocationErrorKind::OutOfMemory);
        assert_eq!(err.class_id, AllocationClassId(5));
        assert_eq!(err.requested_bytes, 8);
        assert_eq!(pool.allocations(), 1);
        assert_eq!(pool.remaining_bytes(), 0);
    }

    #[test]
    fn allocation_error_diagnostics_are_actionable() {
        let err = AllocationError::new(AllocationErrorKind::OutOfMemory, AllocationClassId(19), 48);
        let rendered = err.to_string();
        assert!(rendered.contains("allocation_error(kind=out_of_memory"));
        assert!(rendered.contains("class_id=19"));
        assert!(rendered.contains("requested_bytes=48"));
        assert_eq!(
            allocation_error_hint(AllocationErrorKind::OutOfMemory),
            "Increase per-type pool budget or use max-inline mode for deterministic pre-allocation."
        );
        assert_eq!(
            allocation_error_hint(AllocationErrorKind::PoolUnavailable),
            "Enable/configure a pool provider for inline-then-pool mode, or switch to max-inline mode."
        );
    }
}
