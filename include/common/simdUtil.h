﻿#pragma once
#include "immintrin.h"
#include <random>
#include <ranges>
#include <stdexcept>


inline float sum256f(__m256 vec) {
    __m128 lo = _mm256_castps256_ps128(vec);
    __m128 hi = _mm256_extractf128_ps(vec, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return  _mm_cvtss_f32(sum128);
}

inline int sum256i(__m256i vec) {
    // _mm256 has `hadd` but the two 128 lanes are separated, so it's useless here

    __m128i low = _mm256_castsi256_si128(vec);         // lower 128 bits
    __m128i high = _mm256_extracti128_si256(vec, 1);       // upper 128 bits
    __m128i sum128 = _mm_add_epi32(low, high);

    // 2x horizontal adds to get a single sum:
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    // get first element:
    return _mm_cvtsi128_si32(sum128);
}

inline size_t pad8(size_t num) {
    return (num % 8 == 0) ? num : ((num + 7) / 8) * 8;
}


inline __m256 cvt8epu8_8ps(const uint64_t chars8) {
    // load the 64‑bit value into the lower part of a __m128i and zero the upper part
    __m128i chars = _mm_cvtsi64_si128(static_cast<long long>(chars8));

    __m128i intsLow = _mm_cvtepu8_epi32(chars);          // lower 4 chars -> 32-bit ints
    __m128i charsHigh = _mm_srli_si128(chars, 4);
    __m128i intsHigh = _mm_cvtepu8_epi32(charsHigh);       // upper 4 chars -> 32-bit ints

    // Combine them into one 256-bit integer vector.
    __m256i combined = _mm256_castsi128_si256(intsLow);    // Place intsLow in lower 128 bits.
    combined = _mm256_inserti128_si256(combined, intsHigh, 1); // Insert intsHigh into upper 128 bits.

    // Convert the combined __m256i (8 int32's) to a __m256 of floats.
    return _mm256_cvtepi32_ps(combined);
}

inline __m256 rand256ps(float rMin = 0.f, float rMax = 1.f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(rMin, rMax);

    // Generate 8 random floats
    alignas (32) std::array<float, 8> randomFloats;
    for (float& val : randomFloats) {
        val = dist(gen);
    }

    return _mm256_load_ps(randomFloats.data());
}

inline void seqRan256(__m256* begin, const __m256* end, float rMin = 0.f, float rMax = 1.f) {
    while (begin != end) {
        *begin = rand256ps(rMin, rMax);
        ++begin;
    }
}