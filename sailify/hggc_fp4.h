#ifndef __HGGC_FP4_H__
#define __HGGC_FP4_H__

/*
 * hggc_fp4.h — FP4 (e2m1) support for PPU, standing in for CUDA's cuda_fp4.h.
 *
 * The PPU SDK ships no FP4 header. sailify rewrites #include <cuda_fp4.h> to
 * this file and the __nv_* / __NV_* / cudaRound* spellings to the __hg_* /
 * __HG_* / hggcRound* ones used here, so converted sources keep the CUDA API:
 *   - storage typedefs and __hg_fp4_interpretation_t (__HG_E2M1)
 *   - __hg_cvt_{float,float2,double,double2,halfraw,halfraw2,
 *     bfloat16raw,bfloat16raw2}_to_fp4[x2] and __hg_cvt_fp4[x2]_to_halfraw[2]
 *   - C++ structs __hg_fp4_e2m1, __hg_fp4x2_e2m1, __hg_fp4x4_e2m1
 *
 * Semantics follow cuda_fp4.hpp's software path, which nvcc's
 * cvt.rn.satfinite.e2m1x2 matches: results always saturate to +-6 (e2m1 has
 * no Inf/NaN), NaN converts to +6, underflow keeps the sign (-0), and pairs
 * pack .x in the low nibble. Only hggcRoundNearest (ties to even) and
 * hggcRoundZero are supported, as in CUDA; any other mode rounds toward zero.
 */

/* Pulls in hggc_fp16.h, hggc_bf16.h, hggcRoundMode, __HGGC_ALIGN__ and the
 * half/bf16 -> float helpers; cuda_fp4.h likewise builds on cuda_fp8.h. */
#include "hggc_fp8.h"

#if defined(__HGGCCC__)
#define __HGGC_HOSTDEVICE_FP4__ __host__ __device__
#define __HGGC_HOSTDEVICE_FP4_DECL__ static __host__ __device__ __inline__
#else /* !defined(__HGGCCC__) */
#if defined(__GNUC__)
#define __HGGC_HOSTDEVICE_FP4_DECL__ static __attribute__((unused))
#else
#define __HGGC_HOSTDEVICE_FP4_DECL__ static
#endif /* defined(__GNUC__) */
#define __HGGC_HOSTDEVICE_FP4__
#endif /* defined(__HGGCCC__) */

#if defined(__cplusplus) && __cplusplus >= 201103L
#define __CPP_VERSION_AT_LEAST_11_FP4
#else
/* explicit conversion operators need C++11 */
#define __HGGC_NO_FP4_CONVERSION_OPERATORS__
#endif

typedef __hg_fp8_storage_t __hg_fp4_storage_t;
typedef __hg_fp8_storage_t __hg_fp4x2_storage_t;
typedef __hg_fp8x2_storage_t __hg_fp4x4_storage_t;

typedef enum __hg_fp4_interpretation_t {
    __HG_E2M1,
} __hg_fp4_interpretation_t;

/* ── Narrowing conversions ─────────────────────────────────────────────── */

__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4_storage_t
__hg_cvt_double_to_fp4(const double x,
                       const __hg_fp4_interpretation_t fp4_interpretation,
                       const enum hggcRoundMode rounding) {
    (void)fp4_interpretation;
    unsigned long long int xbits;
#if defined(__HGGCCC__) || (!defined __cplusplus)
    (void)memcpy(&xbits, &x, sizeof(x));
#else
    (void)::std::memcpy(&xbits, &x, sizeof(x));
#endif
    /* 1/2 ulp of the 1-bit e2m1 mantissa, in double mantissa position */
    const unsigned long long int half_ulp = 1ULL << 50U;
    unsigned char sign = (unsigned char)((xbits >> 63U) << 3U);
    const int exp = (int)((xbits >> 52U) & 0x7FFU) - 1023 + 1;
    unsigned char mantissa = (unsigned char)((xbits >> 51U) & 0x1U);
    const unsigned long long int absx = xbits & 0x7FFFFFFFFFFFFFFFULL;
    unsigned char res;

    if (absx <= 0x3FD0000000000000ULL) {          /* |x| <= 0.25: zero */
        res = 0U;
    } else if (absx > 0x4018000000000000ULL) {   /* |x| > 6: saturate */
        if (absx > 0x7FF0000000000000ULL) {
            sign = 0U;                            /* NaN -> +6 */
        }
        res = 0x7U;
    } else if (absx >= 0x3FF0000000000000ULL) {  /* normal, |x| >= 1 */
        res = (unsigned char)((exp << 1) | mantissa);
        const unsigned long long int round = xbits & ((half_ulp << 1U) - 1ULL);
        if (rounding == hggcRoundNearest &&
            ((round > half_ulp) || ((round == half_ulp) && (mantissa & 1U)))) {
            res = (unsigned char)(res + 1U);
        }
    } else {                                      /* denormal, 0.25 < |x| < 1 */
        const int shift = 1 - exp;
        mantissa = (unsigned char)(mantissa | 0x2U);
        res = (unsigned char)(mantissa >> shift);
        if (rounding == hggcRoundNearest) {
            const unsigned long long int round =
                (xbits | (1ULL << 52U)) & ((half_ulp << (shift + 1)) - 1ULL);
            if ((round > (half_ulp << shift)) ||
                ((round == (half_ulp << shift)) && (res & 1U))) {
                res = (unsigned char)(res + 1U);
            }
        }
    }
    return (__hg_fp4_storage_t)(res | sign);
}

/* Same algorithm on the float bits: exact, and avoids fp64 on the device. */
__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4_storage_t
__hg_cvt_float_to_fp4(const float x,
                      const __hg_fp4_interpretation_t fp4_interpretation,
                      const enum hggcRoundMode rounding) {
    (void)fp4_interpretation;
    const unsigned int xbits = __hgfp8_impl_f32_to_bits(x);
    const unsigned int half_ulp = 1U << 21U;
    unsigned char sign = (unsigned char)((xbits >> 31U) << 3U);
    const int exp = (int)((xbits >> 23U) & 0xFFU) - 127 + 1;
    unsigned char mantissa = (unsigned char)((xbits >> 22U) & 0x1U);
    const unsigned int absx = xbits & 0x7FFFFFFFU;
    unsigned char res;

    if (absx <= 0x3E800000U) {                    /* |x| <= 0.25: zero */
        res = 0U;
    } else if (absx > 0x40C00000U) {             /* |x| > 6: saturate */
        if (absx > 0x7F800000U) {
            sign = 0U;                            /* NaN -> +6 */
        }
        res = 0x7U;
    } else if (absx >= 0x3F800000U) {            /* normal, |x| >= 1 */
        res = (unsigned char)((exp << 1) | mantissa);
        const unsigned int round = xbits & ((half_ulp << 1U) - 1U);
        if (rounding == hggcRoundNearest &&
            ((round > half_ulp) || ((round == half_ulp) && (mantissa & 1U)))) {
            res = (unsigned char)(res + 1U);
        }
    } else {                                      /* denormal, 0.25 < |x| < 1 */
        const int shift = 1 - exp;
        mantissa = (unsigned char)(mantissa | 0x2U);
        res = (unsigned char)(mantissa >> shift);
        if (rounding == hggcRoundNearest) {
            const unsigned int round =
                (xbits | (1U << 23U)) & ((half_ulp << (shift + 1)) - 1U);
            if ((round > (half_ulp << shift)) ||
                ((round == (half_ulp << shift)) && (res & 1U))) {
                res = (unsigned char)(res + 1U);
            }
        }
    }
    return (__hg_fp4_storage_t)(res | sign);
}

__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4x2_storage_t
__hg_cvt_double2_to_fp4x2(const double2 x,
                          const __hg_fp4_interpretation_t fp4_interpretation,
                          const enum hggcRoundMode rounding) {
    const unsigned char lo = __hg_cvt_double_to_fp4(x.x, fp4_interpretation, rounding);
    const unsigned char hi = __hg_cvt_double_to_fp4(x.y, fp4_interpretation, rounding);
    return (__hg_fp4x2_storage_t)((hi << 4U) | lo);
}

__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4x2_storage_t
__hg_cvt_float2_to_fp4x2(const float2 x,
                         const __hg_fp4_interpretation_t fp4_interpretation,
                         const enum hggcRoundMode rounding) {
    const unsigned char lo = __hg_cvt_float_to_fp4(x.x, fp4_interpretation, rounding);
    const unsigned char hi = __hg_cvt_float_to_fp4(x.y, fp4_interpretation, rounding);
    return (__hg_fp4x2_storage_t)((hi << 4U) | lo);
}

__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4_storage_t
__hg_cvt_halfraw_to_fp4(const __half_raw x,
                        const __hg_fp4_interpretation_t fp4_interpretation,
                        const enum hggcRoundMode rounding) {
    return __hg_cvt_float_to_fp4(__hgfp8_impl_halfraw_to_f32(x),
                                 fp4_interpretation, rounding);
}

__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4x2_storage_t
__hg_cvt_halfraw2_to_fp4x2(const __half2_raw x,
                           const __hg_fp4_interpretation_t fp4_interpretation,
                           const enum hggcRoundMode rounding) {
    return __hg_cvt_float2_to_fp4x2(__hgfp8_impl_halfraw2_to_f32x2(x),
                                    fp4_interpretation, rounding);
}

__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4_storage_t
__hg_cvt_bfloat16raw_to_fp4(const __ppu_bfloat16_raw x,
                            const __hg_fp4_interpretation_t fp4_interpretation,
                            const enum hggcRoundMode rounding) {
    return __hg_cvt_float_to_fp4(__hgfp8_impl_bf16raw_to_f32(x),
                                 fp4_interpretation, rounding);
}

__HGGC_HOSTDEVICE_FP4_DECL__ __hg_fp4x2_storage_t
__hg_cvt_bfloat16raw2_to_fp4x2(const __ppu_bfloat162_raw x,
                               const __hg_fp4_interpretation_t fp4_interpretation,
                               const enum hggcRoundMode rounding) {
    __ppu_bfloat16_raw raw;
    raw.x = x.x;
    const unsigned char lo = __hg_cvt_bfloat16raw_to_fp4(raw, fp4_interpretation, rounding);
    raw.x = x.y;
    const unsigned char hi = __hg_cvt_bfloat16raw_to_fp4(raw, fp4_interpretation, rounding);
    return (__hg_fp4x2_storage_t)((hi << 4U) | lo);
}

/* ── Widening conversions (exact) ──────────────────────────────────────── */

__HGGC_HOSTDEVICE_FP4_DECL__ __half_raw
__hg_cvt_fp4_to_halfraw(const __hg_fp4_storage_t x,
                        const __hg_fp4_interpretation_t fp4_interpretation) {
    (void)fp4_interpretation;
    /* magnitude codes 0..7 = {0, .5, 1, 1.5, 2, 3, 4, 6} */
    const unsigned int code = (unsigned int)x & 0x7U;
    unsigned int bits;
    if (code == 0U) {
        bits = 0U;
    } else if (code == 1U) {
        bits = 0x3800U;
    } else {
        bits = (((code >> 1U) + 14U) << 10U) | ((code & 1U) << 9U);
    }
    __half_raw res;
    res.x = (unsigned short)(bits | (((unsigned int)x & 0x8U) << 12U));
    return res;
}

__HGGC_HOSTDEVICE_FP4_DECL__ __half2_raw
__hg_cvt_fp4x2_to_halfraw2(const __hg_fp4x2_storage_t x,
                           const __hg_fp4_interpretation_t fp4_interpretation) {
    __half2_raw res;
    res.x = __hg_cvt_fp4_to_halfraw((__hg_fp4_storage_t)(x & 0xFU), fp4_interpretation).x;
    res.y = __hg_cvt_fp4_to_halfraw((__hg_fp4_storage_t)(x >> 4U), fp4_interpretation).x;
    return res;
}

#define __HGGC_FP4_TYPES_EXIST__

/* ── C++ structs ───────────────────────────────────────────────────────── */

#if defined(__cplusplus)

struct __HGGC_ALIGN__(1) __hg_fp4_e2m1 {
  public:
    __hg_fp4_storage_t __x;

#if defined(__CPP_VERSION_AT_LEAST_11_FP4)
    __hg_fp4_e2m1() = default;
#else
    __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1() {}
#endif

#if !defined(__HGGC_NO_FP4_CONVERSIONS__)
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const __half f) {
        __x = __hg_cvt_halfraw_to_fp4(static_cast<__half_raw>(f), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const __ppu_bfloat16 f) {
        __x = __hg_cvt_bfloat16raw_to_fp4(static_cast<__ppu_bfloat16_raw>(f), __HG_E2M1,
                                          hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const float f) {
        __x = __hg_cvt_float_to_fp4(f, __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const double f) {
        __x = __hg_cvt_double_to_fp4(f, __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const unsigned short int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const unsigned int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const unsigned long int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const unsigned long long int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const short int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const long int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4_e2m1(const long long int val) {
        __x = __hg_cvt_float_to_fp4(static_cast<float>(val), __HG_E2M1, hggcRoundNearest);
    }

#if !defined(__HGGC_NO_FP4_CONVERSION_OPERATORS__)
    explicit __HGGC_HOSTDEVICE_FP4__ operator __half() const {
        return static_cast<__half>(__hg_cvt_fp4_to_halfraw(__x, __HG_E2M1));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator float() const {
        return __hgfp8_impl_halfraw_to_f32(__hg_cvt_fp4_to_halfraw(__x, __HG_E2M1));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator __ppu_bfloat16() const {
        return __float2bfloat16_rz(float(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator double() const {
        return static_cast<double>(float(*this));
    }
    /* Every e2m1 value is exact and |v| <= 6, so integral converts only
     * truncate toward zero; unsigned targets clamp negatives to 0. */
    explicit __HGGC_HOSTDEVICE_FP4__ operator unsigned char() const {
        const float f = float(*this);
        return f < 0.0f ? (unsigned char)0U : static_cast<unsigned char>(f);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator unsigned short int() const {
        return static_cast<unsigned short int>(static_cast<unsigned char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator unsigned int() const {
        return static_cast<unsigned int>(static_cast<unsigned char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator unsigned long int() const {
        return static_cast<unsigned long int>(static_cast<unsigned char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator unsigned long long int() const {
        return static_cast<unsigned long long int>(static_cast<unsigned char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator signed char() const {
        return static_cast<signed char>(float(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator char() const {
        if (((char)-1) < (char)0) {
            return static_cast<char>(static_cast<signed char>(*this));
        }
        return static_cast<char>(static_cast<unsigned char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator short int() const {
        return static_cast<short int>(static_cast<signed char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator int() const {
        return static_cast<int>(static_cast<signed char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator long int() const {
        return static_cast<long int>(static_cast<signed char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator long long int() const {
        return static_cast<long long int>(static_cast<signed char>(*this));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator bool() const {
        return (__x & 0x7U) != 0U;
    }
#endif /* !defined(__HGGC_NO_FP4_CONVERSION_OPERATORS__) */
#endif /* !defined(__HGGC_NO_FP4_CONVERSIONS__) */
};

struct __HGGC_ALIGN__(1) __hg_fp4x2_e2m1 {
  public:
    __hg_fp4x2_storage_t __x;

#if defined(__CPP_VERSION_AT_LEAST_11_FP4)
    __hg_fp4x2_e2m1() = default;
#else
    __HGGC_HOSTDEVICE_FP4__ __hg_fp4x2_e2m1() {}
#endif

#if !defined(__HGGC_NO_FP4_CONVERSIONS__)
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4x2_e2m1(const __half2 f) {
        __x = __hg_cvt_halfraw2_to_fp4x2(static_cast<__half2_raw>(f), __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4x2_e2m1(const __ppu_bfloat162 f) {
        __x = __hg_cvt_bfloat16raw2_to_fp4x2(static_cast<__ppu_bfloat162_raw>(f), __HG_E2M1,
                                             hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4x2_e2m1(const float2 f) {
        __x = __hg_cvt_float2_to_fp4x2(f, __HG_E2M1, hggcRoundNearest);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4x2_e2m1(const double2 f) {
        __x = __hg_cvt_double2_to_fp4x2(f, __HG_E2M1, hggcRoundNearest);
    }

#if !defined(__HGGC_NO_FP4_CONVERSION_OPERATORS__)
    explicit __HGGC_HOSTDEVICE_FP4__ operator __half2() const {
        return static_cast<__half2>(__hg_cvt_fp4x2_to_halfraw2(__x, __HG_E2M1));
    }
    explicit __HGGC_HOSTDEVICE_FP4__ operator float2() const {
        return __hgfp8_impl_halfraw2_to_f32x2(__hg_cvt_fp4x2_to_halfraw2(__x, __HG_E2M1));
    }
#endif /* !defined(__HGGC_NO_FP4_CONVERSION_OPERATORS__) */
#endif /* !defined(__HGGC_NO_FP4_CONVERSIONS__) */
};

struct __HGGC_ALIGN__(2) __hg_fp4x4_e2m1 {
  public:
    __hg_fp4x4_storage_t __x;

#if defined(__CPP_VERSION_AT_LEAST_11_FP4)
    __hg_fp4x4_e2m1() = default;
#else
    __HGGC_HOSTDEVICE_FP4__ __hg_fp4x4_e2m1() {}
#endif

#if !defined(__HGGC_NO_FP4_CONVERSIONS__)
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4x4_e2m1(const __half2 flo, const __half2 fhi) {
        const __hg_fp4x2_storage_t rlo =
            __hg_cvt_halfraw2_to_fp4x2(static_cast<__half2_raw>(flo), __HG_E2M1, hggcRoundNearest);
        const __hg_fp4x2_storage_t rhi =
            __hg_cvt_halfraw2_to_fp4x2(static_cast<__half2_raw>(fhi), __HG_E2M1, hggcRoundNearest);
        __x = (__hg_fp4x4_storage_t)(((unsigned int)rhi << 8U) | rlo);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4x4_e2m1(const __ppu_bfloat162 flo,
                                                     const __ppu_bfloat162 fhi) {
        const __hg_fp4x2_storage_t rlo = __hg_cvt_bfloat16raw2_to_fp4x2(
            static_cast<__ppu_bfloat162_raw>(flo), __HG_E2M1, hggcRoundNearest);
        const __hg_fp4x2_storage_t rhi = __hg_cvt_bfloat16raw2_to_fp4x2(
            static_cast<__ppu_bfloat162_raw>(fhi), __HG_E2M1, hggcRoundNearest);
        __x = (__hg_fp4x4_storage_t)(((unsigned int)rhi << 8U) | rlo);
    }
    explicit __HGGC_HOSTDEVICE_FP4__ __hg_fp4x4_e2m1(const float4 f) {
        const float2 flo = {f.x, f.y};
        const float2 fhi = {f.z, f.w};
        const __hg_fp4x2_storage_t rlo = __hg_cvt_float2_to_fp4x2(flo, __HG_E2M1, hggcRoundNearest);
        const __hg_fp4x2_storage_t rhi = __hg_cvt_float2_to_fp4x2(fhi, __HG_E2M1, hggcRoundNearest);
        __x = (__hg_fp4x4_storage_t)(((unsigned int)rhi << 8U) | rlo);
    }

#if !defined(__HGGC_NO_FP4_CONVERSION_OPERATORS__)
    explicit __HGGC_HOSTDEVICE_FP4__ operator float4() const {
        const float2 rlo = __hgfp8_impl_halfraw2_to_f32x2(
            __hg_cvt_fp4x2_to_halfraw2((__hg_fp4x2_storage_t)(__x & 0xFFU), __HG_E2M1));
        const float2 rhi = __hgfp8_impl_halfraw2_to_f32x2(
            __hg_cvt_fp4x2_to_halfraw2((__hg_fp4x2_storage_t)(__x >> 8U), __HG_E2M1));
        const float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
        return res;
    }
#endif /* !defined(__HGGC_NO_FP4_CONVERSION_OPERATORS__) */
#endif /* !defined(__HGGC_NO_FP4_CONVERSIONS__) */
};

#endif /* defined(__cplusplus) */

#endif /* __HGGC_FP4_H__ */
