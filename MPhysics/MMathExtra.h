#pragma once
#include <vector_functions.h>

typedef unsigned int UInt;
typedef int Int;

typedef size_t SizeT;
using Vector2ui = uint2;
using Vector3ui = uint3;
using Vector4ui = uint4;
using Vector2i = int2;
using Vector3i = int3;
using Vector4i = int4;
#define Vector2ui(x, y) make_uint2(x, y)
#define Vector3ui(x, y, z) make_uint3(x, y, z)
#define Vector4ui(x, y, z, w) make_uint4(x, y, z, w)
#define Vector2i(x, y) make_int2(x, y)
#define Vector3i(x, y, z) make_int3(x, y, z)
#define Vector4i(x, y, z, w) make_int4(x, y, z, w)

#define M_E        2.71828182845904523536   // e
#define M_LOG2E    1.44269504088896340736   // log2(e)
#define M_LOG10E   0.434294481903251827651  // log10(e)
#define M_LN2      0.693147180559945309417  // ln(2)
#define M_LN10     2.30258509299404568402   // ln(10)
#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi
#define M_2_SQRTPI 1.12837916709551257390   // 2/sqrt(pi)
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#define M_SQRT1_2  0.707106781186547524401  // 1/sqrt(2)
 
#ifdef DOUBLE_REAL
typedef double Real;
typedef double2 Vector2;
typedef double3 Vector3;
typedef double4 Vector4;
#define Vector2(x,y) make_double2(x,y)
#define Vector3(x,y,z) make_double3(x,y,z)
#define Vector4(x,y,z,w) make_double4(x,y,z,w)
#define EPSILON DBL_EPSILON
#define REAL_MAX DBL_MAX
#define REAL_MIN DBL_MIN
#define UNKNOWN DBL_MAX
#else
typedef float Real;
typedef float2 Vector2;
typedef float3 Vector3;
typedef float4 Vector4;
#define Vector2(x,y) make_float2(x,y)
#define Vector3(x,y,z) make_float3(x,y,z)
#define Vector4(x,y,z,w) make_float4(x,y,z,w)
#define EPSILON FLT_EPSILON
#define REAL_MAX FLT_MAX
#define REAL_MIN FLT_MIN
#define UNKNOWN FLT_MAX
#endif // DOUBLE_REAL