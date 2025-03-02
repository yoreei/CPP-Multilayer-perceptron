#pragma once

#include <chrono>
#include <iterator>
#include <random>
#include <unordered_map>
#include <string>
#include <tuple>

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

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

TimePoint getTimePoint() {
    return std::chrono::high_resolution_clock::now();
}

using Micro = std::chrono::duration<double, std::micro>;
using Milli = std::chrono::duration<double, std::milli>;
using Seconds = std::chrono::duration<double, std::ratio<1>>;

template <typename Unit>
Unit getEpochTime() {
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
BenchId cppBench(const std::string& name) {
    BenchId hashValue = std::hash<std::string>{}(name);
    if (cppBenchMap.find(hashValue) == cppBenchMap.cend()) {
        std::get<std::string>(cppBenchMap[hashValue]) = name;
        std::get<double>(cppBenchMap[hashValue]) = 0.f;
    }
    cppBenchActive[hashValue] = getEpochTime<Milli>().count();
    return hashValue;
}

void cppBenchEnd(BenchId id) {
    if (cppBenchActive.find(id) == cppBenchActive.cend()) {
        throw std::runtime_error("wrong bench id");
    }
    std::get<double>(cppBenchMap[id]) += getEpochTime<Milli>().count() - cppBenchActive[id];
    cppBenchActive.erase(id);
}
void cppBenchPrint() {
    if (cppBenchActive.size() != 0) {
        std::cout << "cppBenchActive is not empty!!!\n";
    }
    for (const auto& e : cppBenchMap) {
        // e.second is a <std::string, double> tuple
        std::cout << std::get<std::string>(e.second) << ": " << std::get<double>(e.second) << "\n";
    }
}
// ^Bench

// Convert a pixel (float) to a character for visualization.
char ASCIIArtFromFloat(float f) {
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

void enableFpExcept() {
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
void randSeq(Iterator begin, Iterator end, std::iter_value_t<Iterator> rMin = 0, std::iter_value_t<Iterator> rMax = 1) {
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

// A CRTP base class that instruments the special member functions.
template <typename Derived>
struct Traceable {
    Traceable() { TraceableLog(std::string(typeid(Derived).name()) + ": default constructed"); }
    Traceable(const Traceable&) { TraceableLog(std::string(typeid(Derived).name()) + ": copy constructed"); }
    Traceable(Traceable&&) { TraceableLog(std::string(typeid(Derived).name()) + ": move constructed"); }
    Traceable& operator=(const Traceable&) {
        TraceableLog(std::string(typeid(Derived).name()) + ": copy assigned");
        return *this;
    }
    Traceable& operator=(Traceable&&) {
        TraceableLog(std::string(typeid(Derived).name()) + ": move assigned");
        return *this;
    }
    ~Traceable() { TraceableLog(std::string(typeid(Derived).name()) + ": destructed"); }
    static void TraceableLog(const std::string& msg) {
        //std::cout << msg << std::endl;
    }
};
