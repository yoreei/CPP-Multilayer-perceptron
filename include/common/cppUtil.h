#pragma once

#include <chrono>
#include <iterator>
#include <random>
#include <unordered_map>
#include <string>
#include <tuple>
#include <iostream>

#ifndef NDEBUG
inline constexpr bool DEBUG = true;
#else
inline constexpr bool DEBUG = false;
#endif


#if defined(DISABLE_INLINING)
#if defined(_MSC_VER)
#define INLINING __declspec(noinline)
#else
#define INLINING __attribute__((noinline))
#endif
#else
#define INLINING 
#endif

#define fastIO ios::sync_with_stdio(false); cin.tie(nullptr);

/************************************
**** TIME
************************************/

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline TimePoint getTimePoint() {
    return std::chrono::high_resolution_clock::now();
}

using Micro = std::chrono::duration<double, std::micro>;
using Milli = std::chrono::duration<double, std::milli>;
using Seconds = std::chrono::duration<double, std::ratio<1>>;

template <typename Unit>
inline Unit getEpochTime() {
    return getTimePoint().time_since_epoch();
}

// vBench
struct IdentityHasher {
    size_t operator()(size_t key) const noexcept {
        return key;
    }
};
using BenchId = size_t;
static std::unordered_map<BenchId, std::tuple<std::string, double>, IdentityHasher> cppBenchMap;
static std::unordered_map<BenchId, double, IdentityHasher> cppBenchActive;

/* not thread-safe! */
inline BenchId cppBench(const std::string& name) {
    BenchId hashValue = std::hash<std::string>{}(name);
    if (cppBenchMap.find(hashValue) == cppBenchMap.cend()) {
        std::get<std::string>(cppBenchMap[hashValue]) = name;
        std::get<double>(cppBenchMap[hashValue]) = 0.f;
    }
    cppBenchActive[hashValue] = getEpochTime<Milli>().count();
    return hashValue;
}

inline void cppBenchEnd(BenchId id) {
    if (cppBenchActive.find(id) == cppBenchActive.cend()) {
        throw std::runtime_error("wrong bench id");
    }
    std::get<double>(cppBenchMap[id]) += getEpochTime<Milli>().count() - cppBenchActive[id];
    cppBenchActive.erase(id);
}

inline void cppBenchPrint() {
    if (cppBenchActive.size() != 0) {
        std::cout << "cppBenchActive is not empty!!!\n";
    }
    for (const auto& e : cppBenchMap) {
        // e.second is a <std::string, double> tuple
        double time = std::get<double>(e.second); // Milli 
        std::string unit = "ms";

        if (time >= 1000.f) {
            time /= 1000.f;
            unit = "s";
        }

        if (time >= 60) {
            time /= 60;
            unit = "m";
        }

        std::cout << std::get<std::string>(e.second) << ": " << time << unit << "\n";
    }
}
// ^Bench

inline uint32_t swapEndian(uint32_t val) {
    return ((val >> 24) & 0xff) |
        ((val << 8) & 0xff0000) |
        ((val >> 8) & 0xff00) |
        ((val << 24) & 0xff000000);
}

// Convert a pixel (float) to a character for visualization.
inline char ASCIIArtFromFloat(float f) {
    if (f > 0.7f) {
        return '#';
    }
    else if (f > 0.4f) {
        return '!';
    }
    else if (f > 0.1f) {
        return '.';
    }
    else if (f >= 0) {
        return ' ';
    }
    else {
        throw std::runtime_error("wrong f");
    }
}

inline void enableFpExcept() {
#if defined(_MSC_VER) && !defined(__clang__)
    // Clear the exception masks for division by zero, invalid operation, and overflow.
    // This means these exceptions will be raised.

    // _EM_INEXACT
    // _EM_UNERFLOW
    // _EM_DENORMAL
    unsigned int current;
    _controlfp_s(&current, 0, 0);
    _controlfp_s(&current, current & ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW), _MCW_EM);
#endif
}

template <typename Iterator>
inline void randSeq(Iterator begin, Iterator end, std::iter_value_t<Iterator> rMin = 0, std::iter_value_t<Iterator> rMax = 1) {
    using T = std::iter_value_t<Iterator>;
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Static Distribution Selection
    using Distribution = std::conditional_t<
        std::is_integral_v<T>, std::uniform_int_distribution<T>,
        std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
        void>>;


    static_assert(!std::is_same_v<Distribution, void>,
        "T must be an integral or floating-point type");

    Distribution dist(rMin, rMax);
    while (begin != end) {
        *begin = dist(gen);
        ++begin;
    }
}
/**************************************
**** LOGGING
**************************************/

template<typename T>
inline void cppLog(const std::span<T>& arr) {
    std::cout << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << "]\n";

}

// A CRTP base class that instruments the special member functions.
template <typename Derived>
struct Traceable {
    inline Traceable() { TraceableLog(std::string(typeid(Derived).name()) + ": default constructed"); }
    inline Traceable(const Traceable&) { TraceableLog(std::string(typeid(Derived).name()) + ": copy constructed"); }
    inline Traceable(Traceable&&) { TraceableLog(std::string(typeid(Derived).name()) + ": move constructed"); }
    inline Traceable& operator=(const Traceable&) {
        TraceableLog(std::string(typeid(Derived).name()) + ": copy assigned");
        return *this;
    }
    inline Traceable& operator=(Traceable&&) {
        TraceableLog(std::string(typeid(Derived).name()) + ": move assigned");
        return *this;
    }
    inline ~Traceable() { TraceableLog(std::string(typeid(Derived).name()) + ": destructed"); }
    inline static void TraceableLog(const std::string& msg) {
        //std::cout << msg << std::endl;
    }
};

/*************************************
**** ALGORITHMS
*************************************/
/*
    std::vector<int> in{ 1,2,3,4,5 };
    std::vector<int> out (2, 0);
    while (nextPermute(in, out)) {
        int m = out[0];
        int n = out[1];
*/
inline bool nextPermute(std::vector<int>& in, std::vector<int>& out) {

    int n = in.size();
    int k = out.size();
    for (int i = 0; i < k; i++)
    {
        out[i] = in[i];
    }
    std::reverse(in.begin() + k, in.end());
    return std::next_permutation(in.begin(), in.end());
}

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdarg.h>

void c_logger(const char *format, ...) {
    va_list args;
    va_start(args, format);
    
    vfprintf(stdout, format, args);
    fflush(stdout);  // Force a flush after printing

    va_end(args);
}

#ifdef __cplusplus
} // extern "C"
#endif

