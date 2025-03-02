#pragma once

#include <chrono>
#include <iterator>
#include <random>

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

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

Time getTime() {
    return std::chrono::high_resolution_clock::now();
}
using Milli = std::chrono::duration<double, std::milli>;
using Seconds = std::chrono::duration<double, std::ratio<1>>;

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
