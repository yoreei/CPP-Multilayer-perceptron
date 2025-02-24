#pragma once

#include <chrono>
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

