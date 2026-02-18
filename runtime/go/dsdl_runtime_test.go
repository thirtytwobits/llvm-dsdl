package dsdlruntime

import (
	"math"
	"math/rand"
	"testing"
)

func getBitRef(buf []byte, offset int) uint8 {
	if offset < 0 {
		return 0
	}
	byteIndex := offset / 8
	if byteIndex >= len(buf) {
		return 0
	}
	return (buf[byteIndex] >> uint(offset%8)) & 1
}

func copyBitsRef(dst []byte, dstOffsetBits, lengthBits int, src []byte, srcOffsetBits int) {
	for i := 0; i < lengthBits; i++ {
		bit := getBitRef(src, srcOffsetBits+i)
		dstBitIndex := dstOffsetBits + i
		if dstBitIndex < 0 || (dstBitIndex/8) >= len(dst) {
			break
		}
		if bit == 1 {
			dst[dstBitIndex/8] |= uint8(1) << uint(dstBitIndex%8)
		} else {
			dst[dstBitIndex/8] &^= uint8(1) << uint(dstBitIndex%8)
		}
	}
}

func TestChooseMin(t *testing.T) {
	if got := ChooseMin(2, 9); got != 2 {
		t.Fatalf("ChooseMin(2,9) = %d, want 2", got)
	}
	if got := ChooseMin(7, 1); got != 1 {
		t.Fatalf("ChooseMin(7,1) = %d, want 1", got)
	}
}

func TestSaturateFragmentBits(t *testing.T) {
	if got := SaturateFragmentBits(2, 0, 9); got != 9 {
		t.Fatalf("SaturateFragmentBits(2,0,9) = %d, want 9", got)
	}
	if got := SaturateFragmentBits(2, 10, 10); got != 6 {
		t.Fatalf("SaturateFragmentBits(2,10,10) = %d, want 6", got)
	}
	if got := SaturateFragmentBits(2, 16, 10); got != 0 {
		t.Fatalf("SaturateFragmentBits(2,16,10) = %d, want 0", got)
	}
}

func TestCopyBitsUnalignedMatchesReference(t *testing.T) {
	src := []byte{0xB6, 0x55, 0xE1}
	dst := []byte{0x00, 0x00, 0x00}
	ref := []byte{0x00, 0x00, 0x00}

	CopyBits(dst, 5, 13, src, 3)
	copyBitsRef(ref, 5, 13, src, 3)

	for i := range dst {
		if dst[i] != ref[i] {
			t.Fatalf("byte[%d] mismatch: got 0x%02X want 0x%02X", i, dst[i], ref[i])
		}
	}
}

func TestCopyBitsRandomizedAgainstReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5EEDC0DE))
	for iter := 0; iter < 1000; iter++ {
		srcLen := rng.Intn(16) + 1
		dstLen := rng.Intn(16) + 1
		src := make([]byte, srcLen)
		dst := make([]byte, dstLen)
		ref := make([]byte, dstLen)
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		for i := range dst {
			b := byte(rng.Intn(256))
			dst[i] = b
			ref[i] = b
		}

		srcOffset := rng.Intn(srcLen * 8)
		dstOffset := rng.Intn(dstLen * 8)
		maxLen := (srcLen * 8) - srcOffset
		if other := (dstLen * 8) - dstOffset; other < maxLen {
			maxLen = other
		}
		if maxLen < 0 {
			maxLen = 0
		}
		length := 0
		if maxLen > 0 {
			length = rng.Intn(maxLen + 1)
		}

		CopyBits(dst, dstOffset, length, src, srcOffset)
		copyBitsRef(ref, dstOffset, length, src, srcOffset)
		for i := range dst {
			if dst[i] != ref[i] {
				t.Fatalf(
					"iter=%d byte[%d] mismatch: got=0x%02X want=0x%02X srcOffset=%d dstOffset=%d length=%d",
					iter,
					i,
					dst[i],
					ref[i],
					srcOffset,
					dstOffset,
					length,
				)
			}
		}
	}
}

func TestGetBitsImplicitZeroExtension(t *testing.T) {
	input := []byte{0xFF}
	var out [2]byte
	GetBits(out[:], input, 0, 16)
	if out[0] != 0xFF || out[1] != 0x00 {
		t.Fatalf("GetBits zero extension mismatch: got [%02X %02X], want [FF 00]", out[0], out[1])
	}
}

func TestSetUxxAndGetU32RoundTrip(t *testing.T) {
	buf := make([]byte, 8)
	if rc := SetUxx(buf, 3, 0xABCDE, 20); rc != DSDL_RUNTIME_SUCCESS {
		t.Fatalf("SetUxx failed with rc=%d", rc)
	}
	got := GetU32(buf, 3, 20)
	if got != 0xABCDE {
		t.Fatalf("GetU32 mismatch: got 0x%X want 0xABCDE", got)
	}
}

func TestSetUxxBufferTooSmall(t *testing.T) {
	buf := make([]byte, 1)
	rc := SetUxx(buf, 7, 0x3, 2)
	if rc != -DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL {
		t.Fatalf("SetUxx rc=%d, want %d", rc, -DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL)
	}
}

func TestSetBitBufferTooSmall(t *testing.T) {
	buf := make([]byte, 1)
	rc := SetBit(buf, 8, true)
	if rc != -DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL {
		t.Fatalf("SetBit rc=%d, want %d", rc, -DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL)
	}
}

func TestSignedGetSignExtension(t *testing.T) {
	negOne3Bit := []byte{0x07} // 0b111 => -1 when interpreted as signed 3-bit
	if got := GetI8(negOne3Bit, 0, 3); got != -1 {
		t.Fatalf("GetI8 sign extension mismatch: got %d want -1", got)
	}
	posThree3Bit := []byte{0x03} // 0b011 => +3
	if got := GetI8(posThree3Bit, 0, 3); got != 3 {
		t.Fatalf("GetI8 positive mismatch: got %d want 3", got)
	}
}

func TestSignedGetSignExtensionWiderTypes(t *testing.T) {
	if got := GetI16([]byte{0x1F}, 0, 5); got != -1 {
		t.Fatalf("GetI16 sign extension mismatch: got %d want -1", got)
	}
	if got := GetI32([]byte{0x00, 0x80}, 0, 16); got != -32768 {
		t.Fatalf("GetI32 sign extension mismatch: got %d want -32768", got)
	}
	if got := GetI64([]byte{0xFF, 0xFF, 0xFF, 0x7F}, 0, 31); got != -1 {
		t.Fatalf("GetI64 sign extension mismatch: got %d want -1", got)
	}
}

func TestFloat16PackUnpackSpecials(t *testing.T) {
	if got := Float16Pack(float32(math.Inf(1))); got != 0x7C00 {
		t.Fatalf("Float16Pack(+Inf)=0x%04X, want 0x7C00", got)
	}
	if got := Float16Pack(float32(math.Inf(-1))); got != 0xFC00 {
		t.Fatalf("Float16Pack(-Inf)=0x%04X, want 0xFC00", got)
	}
	nanPacked := Float16Pack(float32(math.NaN()))
	if (nanPacked&0x7C00) != 0x7C00 || (nanPacked&0x03FF) == 0 {
		t.Fatalf("Float16Pack(NaN)=0x%04X, expected NaN encoding", nanPacked)
	}
	if !math.IsNaN(float64(Float16Unpack(nanPacked))) {
		t.Fatalf("Float16Unpack(NaN encoding) should be NaN")
	}
}

func TestSetGetFloatsRoundTrip(t *testing.T) {
	buf := make([]byte, 16)

	const f32Value float32 = 1.25
	if rc := SetF32(buf, 0, f32Value); rc != DSDL_RUNTIME_SUCCESS {
		t.Fatalf("SetF32 failed with rc=%d", rc)
	}
	if got := GetF32(buf, 0); got != f32Value {
		t.Fatalf("GetF32 mismatch: got %f want %f", got, f32Value)
	}

	const f64Value float64 = -1234.5
	if rc := SetF64(buf, 32, f64Value); rc != DSDL_RUNTIME_SUCCESS {
		t.Fatalf("SetF64 failed with rc=%d", rc)
	}
	if got := GetF64(buf, 32); got != f64Value {
		t.Fatalf("GetF64 mismatch: got %f want %f", got, f64Value)
	}
}

func TestGetUnsignedOutOfRangeIsZeroExtended(t *testing.T) {
	buf := []byte{0xAA}
	if got := GetU16(buf, 16, 16); got != 0 {
		t.Fatalf("GetU16 out-of-range mismatch: got 0x%X want 0", got)
	}
	if got := GetU64(buf, 64, 64); got != 0 {
		t.Fatalf("GetU64 out-of-range mismatch: got 0x%X want 0", got)
	}
}
