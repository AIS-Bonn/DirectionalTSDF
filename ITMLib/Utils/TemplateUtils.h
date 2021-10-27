//
// Created by Malte Splietker on 22.10.21.
//

#pragma once

/// Macro to dynamically dispatch member functions with templates. Usage: DispatchMemberFunction(function, boolean)(params...);
#define DispatchMemberFunction(func, b) [this](auto&&... args){if (b) return func<true>(args...); else return func<false>(args...);}

/// Macro to dynamically dispatch functions with templates. Usage: DispatchFunction(function, boolean)(params...);
#define DispatchFunction(func, b) [](auto&&... args){if (b) return func<true>(args...); else return func<false>(args...);}