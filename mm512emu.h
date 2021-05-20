#ifndef MM512EMU_INCLUDED
#define MM512EMU_INCLUDED

/*
Include this header if __AVX512F__ is not defined for emulation, requires AVX2+FMA.

#ifndef __AVX512F__
#include "avx512emu.h"
#endif

Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <immintrin.h>

#ifdef _MSC_VER
    #define mm512emu_forceinline_spec __forceinline
    #define mm512emu_forceinline static __forceinline
#else
    #define mm512emu_forceinline_spec inline __attribute__((always_inline))
    #define mm512emu_forceinline static inline __attribute__((always_inline))
#endif

namespace mm512emu {

    // Type defintions

    struct m512 {
        __m256 lo, hi;
    };

    struct m512d {
        __m256d lo, hi;
    };

    struct m512i {
        __m256i lo, hi;
    };

    struct mask8 {
        __m256i lo, hi;
    };

    // Internal

    namespace imp {
        __m256i not_si256(__m256i a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
        __m256i cmpgt_epu64(__m256i a, __m256i b) {
            __m256i bias = _mm256_set1_epi64x(INT64_MIN);
            return _mm256_cmpgt_epi64(_mm256_add_epi64(a, bias), _mm256_add_epi64(b, bias));
        }
    }

    // -- *_ps()

    // Load/store

    mm512emu_forceinline m512 setzero_ps() { return { _mm256_setzero_ps(), _mm256_setzero_ps() }; }
    mm512emu_forceinline m512 set1_ps(float a) { return { _mm256_set1_ps(a), _mm256_set1_ps(a) }; }
    mm512emu_forceinline m512 broadcastss_ps(__m128 a) { return { _mm256_broadcastss_ps(a), _mm256_broadcastss_ps(a) }; }

    mm512emu_forceinline m512 load_ps(void const* mem_addr) {
        return { _mm256_load_ps((const float*)mem_addr), _mm256_load_ps((const float*)mem_addr + 8) };
    }
    mm512emu_forceinline m512 loadu_ps(void const* mem_addr) {
        return { _mm256_loadu_ps((const float*)mem_addr), _mm256_loadu_ps((const float*)mem_addr + 8) };
    }
    mm512emu_forceinline void store_ps(void* mem_addr, m512 a) {
        _mm256_store_ps((float*)mem_addr, a.lo);
        _mm256_store_ps((float*)mem_addr + 8, a.hi);
    }
    mm512emu_forceinline void storeu_ps(void* mem_addr, m512 a) {
        _mm256_storeu_ps((float*)mem_addr, a.lo);
        _mm256_storeu_ps((float*)mem_addr + 8, a.hi);
    }
    mm512emu_forceinline void stream_ps(void* mem_addr, m512 a) {
        _mm256_stream_ps((float*)mem_addr, a.lo);
        _mm256_stream_ps((float*)mem_addr + 8, a.hi);
    }

    // Arithmetic

    mm512emu_forceinline m512 add_ps(m512 a, m512 b) { return { _mm256_add_ps(a.lo, b.lo), _mm256_add_ps(a.hi, b.hi) }; }
    mm512emu_forceinline m512 sub_ps(m512 a, m512 b) { return { _mm256_sub_ps(a.lo, b.lo), _mm256_sub_ps(a.hi, b.hi) }; }
    mm512emu_forceinline m512 mul_ps(m512 a, m512 b) { return { _mm256_mul_ps(a.lo, b.lo), _mm256_mul_ps(a.hi, b.hi) }; }
    mm512emu_forceinline m512 div_ps(m512 a, m512 b) { return { _mm256_div_ps(a.lo, b.lo), _mm256_div_ps(a.hi, b.hi) }; }
    mm512emu_forceinline m512 fmadd_ps(m512 a, m512 b, m512 c) { return { _mm256_fmadd_ps(a.lo, b.lo, c.lo), _mm256_fmadd_ps(a.hi, b.hi, c.hi) }; }
    mm512emu_forceinline m512 fmsub_pd(m512 a, m512 b, m512 c) { return { _mm256_fmsub_ps(a.lo, b.lo, c.lo), _mm256_fmsub_ps(a.hi, b.hi, c.hi) }; }
    mm512emu_forceinline m512 min_ps(m512 a, m512 b) { return { _mm256_min_ps(a.lo, b.lo), _mm256_min_ps(a.hi, b.hi) }; }
    mm512emu_forceinline m512 max_ps(m512 a, m512 b) { return { _mm256_max_ps(a.lo, b.lo), _mm256_max_ps(a.hi, b.hi) }; }

    // Shuffle

    template <int Imm8> mm512emu_forceinline m512 permute_ps(m512 a) { return { _mm256_permute_ps(a.lo, Imm8&0xf), _mm256_permute_ps(a.hi, Imm8>>4) }; }
    mm512emu_forceinline __m128 castps512_ps128(m512 a) { return _mm256_castps256_ps128(a.lo); }
    mm512emu_forceinline __m256 castps512_ps256(m512 a) { return a.lo; }
    mm512emu_forceinline m512 castps256_ps512(__m256 a) { return { a, _mm256_setzero_ps() }; }
    template <int Imm8> mm512emu_forceinline __m256 extractf32x8_ps(m512 a) { return (Imm8&1) ? a.hi : a.lo; }
    template <int Imm8> mm512emu_forceinline m512 insertf32x8(m512 a, __m256d b) { return (Imm8&1) ? m512{ a.lo, b } : m512{ b, a.hi }; }
    mm512emu_forceinline m512 unpacklo_ps(m512 a, m512 b) { return { _mm256_unpacklo_ps(a.lo, b.lo), _mm256_unpacklo_ps(a.hi, b.hi) }; }
    mm512emu_forceinline m512 unpackhi_ps(m512 a, m512 b) { return { _mm256_unpackhi_ps(a.lo, b.lo), _mm256_unpackhi_ps(a.hi, b.hi) }; }

    // Conversion

    mm512emu_forceinline __m256 cvtpd_ps(m512d a) { return _mm256_insertf128_ps(_mm256_castpd_ps(a.lo), _mm256_castps256_ps128(_mm256_castpd_ps(a.hi)), 1); }

    // -- *_pd()

    // Load/store

    mm512emu_forceinline m512d setzero_pd() { return { _mm256_setzero_pd(), _mm256_setzero_pd() }; }
    mm512emu_forceinline m512d set1_pd(double a) { return { _mm256_set1_pd(a), _mm256_set1_pd(a) }; }
    mm512emu_forceinline m512d broadcastsd_pd(__m128d a) { return { _mm256_broadcastsd_pd(a), _mm256_broadcastsd_pd(a) }; }

    mm512emu_forceinline m512d load_pd(void const* mem_addr) {
        return { _mm256_load_pd((const double*)mem_addr), _mm256_load_pd((const double*)mem_addr + 4) };
    }
    mm512emu_forceinline m512d loadu_pd(void const* mem_addr) {
        return { _mm256_loadu_pd((const double*)mem_addr), _mm256_loadu_pd((const double*)mem_addr + 4) };
    }
    mm512emu_forceinline void store_pd(void* mem_addr, m512d a) {
        _mm256_store_pd((double*)mem_addr, a.lo);
        _mm256_store_pd((double*)mem_addr + 4, a.hi);
    }
    mm512emu_forceinline void storeu_pd(void* mem_addr, m512d a) {
        _mm256_storeu_pd((double*)mem_addr, a.lo);
        _mm256_storeu_pd((double*)mem_addr + 4, a.hi);
    }
    mm512emu_forceinline void stream_pd(void* mem_addr, m512d a) {
        _mm256_stream_pd((double*)mem_addr, a.lo);
        _mm256_stream_pd((double*)mem_addr + 4, a.hi);
    }

    // Arithmetic

    mm512emu_forceinline m512d add_pd(m512d a, m512d b) { return { _mm256_add_pd(a.lo, b.lo), _mm256_add_pd(a.hi, b.hi) }; }
    mm512emu_forceinline m512d sub_pd(m512d a, m512d b) { return { _mm256_sub_pd(a.lo, b.lo), _mm256_sub_pd(a.hi, b.hi) }; }
    mm512emu_forceinline m512d mul_pd(m512d a, m512d b) { return { _mm256_mul_pd(a.lo, b.lo), _mm256_mul_pd(a.hi, b.hi) }; }
    mm512emu_forceinline m512d div_pd(m512d a, m512d b) { return { _mm256_div_pd(a.lo, b.lo), _mm256_div_pd(a.hi, b.hi) }; }
    mm512emu_forceinline m512d fmadd_pd(m512d a, m512d b, m512d c) { return { _mm256_fmadd_pd(a.lo, b.lo, c.lo), _mm256_fmadd_pd(a.hi, b.hi, c.hi) }; }
    mm512emu_forceinline m512d fmsub_pd(m512d a, m512d b, m512d c) { return { _mm256_fmsub_pd(a.lo, b.lo, c.lo), _mm256_fmsub_pd(a.hi, b.hi, c.hi) }; }
    mm512emu_forceinline m512d min_pd(m512d a, m512d b) { return { _mm256_min_pd(a.lo, b.lo), _mm256_min_pd(a.hi, b.hi) }; }
    mm512emu_forceinline m512d max_pd(m512d a, m512d b) { return { _mm256_max_pd(a.lo, b.lo), _mm256_max_pd(a.hi, b.hi) }; }

    // Masks

    template <int Imm8> mask8 cmp_pd_mask(m512d a, m512d b) { return { _mm256_castpd_si256(_mm256_cmp_pd(a.lo, b.lo, Imm8)), _mm256_castpd_si256(_mm256_cmp_pd(a.hi, b.hi, Imm8)) }; }
    m512d mask_mov_pd(m512d src, mask8 mask, m512d a) { return { _mm256_blendv_pd(src.lo, a.lo, _mm256_castsi256_pd(mask.lo)), _mm256_blendv_pd(src.hi, a.hi, _mm256_castsi256_pd(mask.hi)) }; }

    // Shuffle

    template <int Imm8> mm512emu_forceinline m512d permute_pd(m512d a) { return { _mm256_permute_pd(a.lo, Imm8&0xf), _mm256_permute_pd(a.hi, Imm8>>4) }; }
    template <int Imm8> mm512emu_forceinline m512d permutex_pd(m512d a) { return { _mm256_permute4x64_pd(a.lo, Imm8), _mm256_permute4x64_pd(a.hi, Imm8) }; }
    mm512emu_forceinline __m128d castpd512_pd128(m512d a) { return _mm256_castpd256_pd128(a.lo); }
    mm512emu_forceinline __m256d castpd512_pd256(m512d a) { return a.lo; }
    mm512emu_forceinline m512d castpd256_pd512(__m256d a) { return { a, _mm256_setzero_pd() }; }
    template <int Imm8> mm512emu_forceinline __m256d extractf64x4_pd(m512d a) { return (Imm8&1) ? a.hi : a.lo; }
    template <int Imm8> mm512emu_forceinline m512d insertf64x4(m512d a, __m256d b) { return (Imm8&1) ? m512d{ a.lo, b } : m512d{ b, a.hi }; }
    mm512emu_forceinline m512d unpacklo_pd(m512d a, m512d b) { return { _mm256_unpacklo_pd(a.lo, b.lo), _mm256_unpacklo_pd(a.hi, b.hi) }; }
    mm512emu_forceinline m512d unpackhi_pd(m512d a, m512d b) { return { _mm256_unpackhi_pd(a.lo, b.lo), _mm256_unpackhi_pd(a.hi, b.hi) }; }

    // Conversion

    mm512emu_forceinline m512d cvtps_pd(__m256 a) { return { _mm256_cvtps_pd(_mm256_castps256_ps128(a)), _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1)) }; }

    // -- *_si512/epi/etc()

    // Load/store

    mm512emu_forceinline m512i load_si512(const void* mem_addr) { return { _mm256_load_si256((const __m256i*)mem_addr), _mm256_load_si256((const __m256i*)mem_addr + 1) }; }
    mm512emu_forceinline m512i loadu_si512(const void* mem_addr) { return { _mm256_loadu_si256((const __m256i*)mem_addr), _mm256_loadu_si256((const __m256i*)mem_addr + 1) }; }
    mm512emu_forceinline m512i stream_load_si512(const void* mem_addr) { return { _mm256_stream_load_si256((const __m256i*)mem_addr), _mm256_stream_load_si256((const __m256i*)mem_addr + 1) }; }
    mm512emu_forceinline m512i set1_epi64(int64_t a) { return { _mm256_set1_epi64x(a), _mm256_set1_epi64x(a) }; }

    inline void mask_compressstoreu_epi64(void *base_addr, mask8 k, m512i a) {
        alignas(32) int64_t tmp[8];
        uint64_t mask = (uint64_t)(uint32_t)_mm256_movemask_epi8(k.lo) | (uint64_t)(uint32_t)_mm256_movemask_epi8(k.hi) << 32u;
        _mm256_store_si256((__m256i*)(tmp + 0), a.lo);
        _mm256_store_si256((__m256i*)(tmp + 4), a.hi);
        int64_t *dst = (int64_t*)base_addr;
        for (size_t i = 0; i < 8; i++) {
            if (mask & 0x80) {
                *dst++ = tmp[i];
            }
            mask >>= 8;
        }
    }

    // Comparison

    template <int Imm8> mask8 cmp_epu64_mask(m512i a, m512i b) = delete;
    template <> mm512emu_forceinline_spec mask8 cmp_epu64_mask<_MM_CMPINT_EQ>(m512i a, m512i b) { return { _mm256_cmpeq_epi64(a.lo, b.lo), _mm256_cmpeq_epi64(a.hi, b.hi) }; }
    template <> mm512emu_forceinline_spec mask8 cmp_epu64_mask<_MM_CMPINT_NE>(m512i a, m512i b) { return { imp::not_si256(_mm256_cmpeq_epi64(a.lo, b.lo)), imp::not_si256(_mm256_cmpeq_epi64(a.hi, b.hi)) }; }
    template <> mm512emu_forceinline_spec mask8 cmp_epu64_mask<_MM_CMPINT_LT>(m512i a, m512i b) { return { imp::cmpgt_epu64(b.lo, a.lo), imp::cmpgt_epu64(b.hi, a.hi) }; }
    template <> mm512emu_forceinline_spec mask8 cmp_epu64_mask<_MM_CMPINT_NLT>(m512i a, m512i b) { return { imp::not_si256(imp::cmpgt_epu64(b.lo, a.lo)), imp::not_si256(imp::cmpgt_epu64(b.hi, a.hi)) }; }
    template <> mm512emu_forceinline_spec mask8 cmp_epu64_mask<_MM_CMPINT_LE>(m512i a, m512i b) { return { imp::not_si256(imp::cmpgt_epu64(a.lo, b.lo)), imp::not_si256(imp::cmpgt_epu64(a.hi, b.hi)) }; }
    template <> mm512emu_forceinline_spec mask8 cmp_epu64_mask<_MM_CMPINT_NLE>(m512i a, m512i b) { return { imp::cmpgt_epu64(a.lo, b.lo), imp::cmpgt_epu64(a.hi, b.hi) }; }

    // Conversion

    m512 castsi512_ps(m512i a) { return { _mm256_castsi256_ps(a.lo), _mm256_castsi256_ps(a.hi) }; }
    m512d castsi512_pd(m512i a) { return { _mm256_castsi256_pd(a.lo), _mm256_castsi256_pd(a.hi) }; }

    // -- Mask

    uint32_t cvtmask8_u32(mask8 a) { return _pext_u32(_mm256_movemask_epi8(a.lo), 0x80808080u) | _pext_u32(_mm256_movemask_epi8(a.hi), 0x80808080u) << 4; }
    mask8 knot_mask8(mask8 a) { return { imp::not_si256(a.lo), imp::not_si256(a.hi) }; }
    mask8 kand_mask8(mask8 a, mask8 b) { return { _mm256_and_si256(a.lo, b.lo), _mm256_and_si256(a.hi, b.hi) }; }
    mask8 kor_mask8(mask8 a, mask8 b) { return { _mm256_or_si256(a.lo, b.lo), _mm256_or_si256(a.hi, b.hi) }; }
    mask8 kandn_mask8(mask8 a, mask8 b) { return { _mm256_andnot_si256(a.lo, b.lo), _mm256_andnot_si256(a.hi, b.hi) }; }
}


#undef __m512
#define __m512 mm512emu::m512

#undef __m512d
#define __m512d mm512emu::m512d

#undef __m512i
#define __m512i mm512emu::m512i

#undef __mmask8
#define __mmask8 mm512emu::mask8

#undef _mm512_setzero_ps
#define _mm512_setzero_ps() mm512emu::setzero_ps()
#undef _mm512_set1_ps
#define _mm512_set1_ps(a) mm512emu::set1_ps((a))
#undef _mm512_broadcastss_ps
#define _mm512_broadcastss_ps(a) mm512emu::broadcastss_ps((a))
#undef _mm512_load_ps
#define _mm512_load_ps(mem_addr) mm512emu::load_ps((mem_addr))
#undef _mm512_loadu_ps
#define _mm512_loadu_ps(mem_addr) mm512emu::loadu_ps((mem_addr))
#undef _mm512_store_ps
#define _mm512_store_ps(mem_addr, a) mm512emu::store_ps((mem_addr), (a))
#undef _mm512_storeu_ps
#define _mm512_storeu_ps(mem_addr, a) mm512emu::storeu_ps((mem_addr), (a))
#undef _mm512_stream_ps
#define _mm512_stream_ps(mem_addr, a) mm512emu::stream_ps((mem_addr), (a))
#undef _mm512_add_ps
#define _mm512_add_ps(a, b) mm512emu::add_ps((a), (b))
#undef _mm512_sub_ps
#define _mm512_sub_ps(a, b) mm512emu::sub_ps((a), (b))
#undef _mm512_mul_ps
#define _mm512_mul_ps(a, b) mm512emu::mul_ps((a), (b))
#undef _mm512_div_ps
#define _mm512_div_ps(a, b) mm512emu::div_ps((a), (b))
#undef _mm512_fmadd_ps
#define _mm512_fmadd_ps(a, b, c) mm512emu::fmadd_ps((a), (b), (c))
#undef _mm512_fmsub_ps
#define _mm512_fmsub_ps(a, b, c) mm512emu::fmsub_ps((a), (b), (c))
#undef _mm512_min_ps
#define _mm512_min_ps(a, b) mm512emu::min_ps((a), (b))
#undef _mm512_max_ps
#define _mm512_max_ps(a, b) mm512emu::max_ps((a), (b))
#undef _mm512_castp512_pd256
#define _mm512_castps512_ps256(a) mm512emu::castps512_ps256((a))
#undef _mm512_castps256_ps512
#define _mm512_castps256_ps512(a) mm512emu::castps256_ps512((a))
#undef _mm512_permute_ps
#define _mm512_permute_ps(a, imm8) mm512emu::permute_ps<(imm8)>((a))
#undef _mm512_extractf32x8_ps
#define _mm512_extractf32x8_ps(a, imm8) mm512emu::extractf32x8_ps<(imm8)>((a))
#undef _mm512_insertf32x8
#define _mm512_insertf32x8(a, b, imm8) mm512emu::insertf32x8<(imm8)>((a), (b))
#undef _mm512_unpacklo_ps
#define _mm512_unpacklo_ps(a, b) mm512emu::unpacklo_ps((a), (b))
#undef _mm512_unpackhi_ps
#define _mm512_unpackhi_ps(a, b) mm512emu::unpackhi_ps((a), (b))
#undef _mm512_cvtpd_ps
#define _mm512_cvtpd_ps(a) mm512emu::cvtpd_ps((a))

#undef _mm512_setzero_pd
#define _mm512_setzero_pd() mm512emu::setzero_pd()
#undef _mm512_set1_pd
#define _mm512_set1_pd(a) mm512emu::set1_pd((a))
#undef _mm512_broadcastsd_pd
#define _mm512_broadcastsd_pd(a) mm512emu::broadcastsd_pd((a))
#undef _mm512_load_pd
#define _mm512_load_pd(mem_addr) mm512emu::load_pd((mem_addr))
#undef _mm512_loadu_pd
#define _mm512_loadu_pd(mem_addr) mm512emu::loadu_pd((mem_addr))
#undef _mm512_store_pd
#define _mm512_store_pd(mem_addr, a) mm512emu::store_pd((mem_addr), (a))
#undef _mm512_storeu_pd
#define _mm512_storeu_pd(mem_addr, a) mm512emu::storeu_pd((mem_addr), (a))
#undef _mm512_stream_pd
#define _mm512_stream_pd(mem_addr, a) mm512emu::stream_pd((mem_addr), (a))
#undef _mm512_add_pd
#define _mm512_add_pd(a, b) mm512emu::add_pd((a), (b))
#undef _mm512_sub_pd
#define _mm512_sub_pd(a, b) mm512emu::sub_pd((a), (b))
#undef _mm512_mul_pd
#define _mm512_mul_pd(a, b) mm512emu::mul_pd((a), (b))
#undef _mm512_div_pd
#define _mm512_div_pd(a, b) mm512emu::div_pd((a), (b))
#undef _mm512_fmadd_pd
#define _mm512_fmadd_pd(a, b, c) mm512emu::fmadd_pd((a), (b), (c))
#undef _mm512_fmsub_pd
#define _mm512_fmsub_pd(a, b, c) mm512emu::fmsub_pd((a), (b), (c))
#undef _mm512_min_pd
#define _mm512_min_pd(a, b) mm512emu::min_pd((a), (b))
#undef _mm512_max_pd
#define _mm512_max_pd(a, b) mm512emu::max_pd((a), (b))
#undef _mm512_cmp_pd_mask
#define _mm512_cmp_pd_mask(a, b, imm8) mm512emu::cmp_pd_mask<(imm8)>((a), (b))
#undef _mm512_mask_mov_pd
#define _mm512_mask_mov_pd(src, k, a) mm512emu::mask_mov_pd((src), (k), (a))
#undef _mm512_permute_pd
#define _mm512_permute_pd(a, imm8) mm512emu::permute_pd<(imm8)>((a))
#undef _mm512_permutex_pd
#define _mm512_permutex_pd(a, imm8) mm512emu::permutex_pd<(imm8)>((a))
#undef _mm512_castpd512_pd128
#define _mm512_castpd512_pd128(a) mm512emu::castpd512_pd128((a))
#undef _mm512_castpd512_pd256
#define _mm512_castpd512_pd256(a) mm512emu::castpd512_pd256((a))
#undef _mm512_castpd256_pd512
#define _mm512_castpd256_pd512(a) mm512emu::castpd256_pd512((a))
#undef _mm512_extractf64x4_pd
#define _mm512_extractf64x4_pd(a, imm8) mm512emu::extractf64x4_pd<(imm8)>((a))
#undef _mm512_insertf64x4
#define _mm512_insertf64x4(a, b, imm8) mm512emu::insertf64x4<(imm8)>((a), (b))
#undef _mm512_unpacklo_pd
#define _mm512_unpacklo_pd(a, b) mm512emu::unpacklo_pd((a), (b))
#undef _mm512_unpackhi_pd
#define _mm512_unpackhi_pd(a, b) mm512emu::unpackhi_pd((a), (b))
#undef _mm512_cvtps_pd
#define _mm512_cvtps_pd(a) mm512emu::cvtps_pd((a))

#undef _mm512_load_si512
#define _mm512_load_si512(a) mm512emu:load_si512((a))
#undef _mm512_loadu_si512
#define _mm512_loadu_si512(a) mm512emu:loadu_si512((a))
#undef _mm512_stream_load_si512
#define _mm512_stream_load_si512(a) mm512emu::stream_load_si512((a))
#undef _mm512_load_epi32
#define _mm512_load_epi32(a) mm512emu::load_si512((a))
#undef _mm512_loadu_epi32
#define _mm512_loadu_epi32(a) mm512emu::loadu_si512((a))
#undef _mm512_load_epi64
#define _mm512_load_epi64(a) mm512emu::load_si512((a))
#undef _mm512_loadu_epi64
#define _mm512_loadu_epi64(a) mm512emu::loadu_si512((a))
#undef _mm512_set1_epi64
#define _mm512_set1_epi64(a) mm512emu::set1_epi64((a))
#undef _mm512_mask_compressstoreu_epi64
#define _mm512_mask_compressstoreu_epi64(base_addr, k, a) mm512emu::mask_compressstoreu_epi64((base_addr), (k), (a))
#undef _mm512_cmp_epu64_mask
#define _mm512_cmp_epu64_mask(a, b, imm8) mm512emu::cmp_epu64_mask<(imm8)>((a), (b))
#undef _mm512_castsi512_ps
#define _mm512_castsi512_ps(a) mm512emu::castsi512_ps((a))
#undef _mm512_castsi512_pd
#define _mm512_castsi512_pd(a) mm512emu::castsi512_pd((a))


#undef _cvtmask8_u32
#define _cvtmask8_u32(a) mm512emu::cvtmask8_u32((a))
#undef _knot_mask8
#define _knot_mask8(a) mm512emu::knot_mask8((a))
#undef _kand_mask8
#define _kand_mask8(a, b) mm512emu::kand_mask8((a), (b))
#undef _kor_mask8
#define _kor_mask8(a, b) mm512emu::kor_mask8((a), (b))
#undef _kandn_mask8
#define _kandn_mask8(a, b) mm512emu::kandn_mask8((a), (b))

#endif
