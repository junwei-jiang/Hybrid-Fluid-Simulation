#pragma once
#include "MMathExtra.h"
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Real fminfM(Real a, Real b)
{
	return a < b ? a : b;
}

inline __host__ __device__ Real fmaxfM(Real a, Real b)
{
	return a > b ? a : b;
}

inline __host__ __device__ Int maxM(Int a, Int b)
{
	return a > b ? a : b;
}

inline __host__ __device__ Int minM(Int a, Int b)
{
	return a < b ? a : b;
}

inline __host__ __device__ Real rsqrtM(Real x)
{
	return 1.0f / sqrt(x);
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 operator-(Vector2 &a)
{
	return Vector2(-a.x, -a.y);
}
inline __host__ __device__ Vector2i operator-(Vector2i &a)
{
	return Vector2i(-a.x, -a.y);
}
inline __host__ __device__ Vector3 operator-(Vector3 &a)
{
	return Vector3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ Vector3i operator-(Vector3i &a)
{
	return Vector3i(-a.x, -a.y, -a.z);
}
inline __host__ __device__ Vector4 operator-(Vector4 &a)
{
	return Vector4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ Vector4i operator-(Vector4i &a)
{
	return Vector4i(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 operator+(Vector2 a, Vector2 b)
{
	return Vector2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(Vector2 &a, Vector2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ Vector2 operator+(Vector2 a, Real b)
{
	return Vector2(a.x + b, a.y + b);
}
inline __host__ __device__ Vector2 operator+(Real b, Vector2 a)
{
	return Vector2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(Vector2 &a, Real b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ Vector2i operator+(Vector2i a, Vector2i b)
{
	return Vector2i(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(Vector2i &a, Vector2i b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ Vector2i operator+(Vector2i a, Int b)
{
	return Vector2i(a.x + b, a.y + b);
}
inline __host__ __device__ Vector2i operator+(Int b, Vector2i a)
{
	return Vector2i(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(Vector2i &a, Int b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ Vector2ui operator+(Vector2ui a, Vector2ui b)
{
	return Vector2ui(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(Vector2ui &a, Vector2ui b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ Vector2ui operator+(Vector2ui a, UInt b)
{
	return Vector2ui(a.x + b, a.y + b);
}
inline __host__ __device__ Vector2ui operator+(UInt b, Vector2ui a)
{
	return Vector2ui(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(Vector2ui &a, UInt b)
{
	a.x += b;
	a.y += b;
}


inline __host__ __device__ Vector3 operator+(Vector3 a, Vector3 b)
{
	return Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(Vector3 &a, Vector3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ Vector3 operator+(Vector3 a, Real b)
{
	return Vector3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(Vector3 &a, Real b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ Vector3i operator+(Vector3i a, Vector3i b)
{
	return Vector3i(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(Vector3i &a, Vector3i b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ Vector3i operator+(Vector3i a, Int b)
{
	return Vector3i(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(Vector3i &a, Int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ Vector3ui operator+(Vector3ui a, Vector3ui b)
{
	return Vector3ui(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(Vector3ui &a, Vector3ui b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ Vector3ui operator+(Vector3ui a, UInt b)
{
	return Vector3ui(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(Vector3ui &a, UInt b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ Vector3i operator+(Int b, Vector3i a)
{
	return Vector3i(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ Vector3ui operator+(UInt b, Vector3ui a)
{
	return Vector3ui(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ Vector3 operator+(Real b, Vector3 a)
{
	return Vector3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ Vector4 operator+(Vector4 a, Vector4 b)
{
	return Vector4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(Vector4 &a, Vector4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ Vector4 operator+(Vector4 a, Real b)
{
	return Vector4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ Vector4 operator+(Real b, Vector4 a)
{
	return Vector4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(Vector4 &a, Real b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ Vector4i operator+(Vector4i a, Vector4i b)
{
	return Vector4i(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(Vector4i &a, Vector4i b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ Vector4i operator+(Vector4i a, Int b)
{
	return Vector4i(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ Vector4i operator+(Int b, Vector4i a)
{
	return Vector4i(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(Vector4i &a, Int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ Vector4ui operator+(Vector4ui a, Vector4ui b)
{
	return Vector4ui(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(Vector4ui &a, Vector4ui b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ Vector4ui operator+(Vector4ui a, UInt b)
{
	return Vector4ui(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ Vector4ui operator+(UInt b, Vector4ui a)
{
	return Vector4ui(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(Vector4ui &a, UInt b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 operator-(Vector2 a, Vector2 b)
{
	return Vector2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(Vector2 &a, Vector2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ Vector2 operator-(Vector2 a, Real b)
{
	return Vector2(a.x - b, a.y - b);
}
inline __host__ __device__ Vector2 operator-(Real b, Vector2 a)
{
	return Vector2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(Vector2 &a, Real b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ Vector2i operator-(Vector2i a, Vector2i b)
{
	return Vector2i(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(Vector2i &a, Vector2i b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ Vector2i operator-(Vector2i a, Int b)
{
	return Vector2i(a.x - b, a.y - b);
}
inline __host__ __device__ Vector2i operator-(Int b, Vector2i a)
{
	return Vector2i(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(Vector2i &a, Int b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ Vector2ui operator-(Vector2ui a, Vector2ui b)
{
	return Vector2ui(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(Vector2ui &a, Vector2ui b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ Vector2ui operator-(Vector2ui a, UInt b)
{
	return Vector2ui(a.x - b, a.y - b);
}
inline __host__ __device__ Vector2ui operator-(UInt b, Vector2ui a)
{
	return Vector2ui(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(Vector2ui &a, UInt b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ Vector3 operator-(Vector3 a, Vector3 b)
{
	return Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(Vector3 &a, Vector3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ Vector3 operator-(Vector3 a, Real b)
{
	return Vector3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ Vector3 operator-(Real b, Vector3 a)
{
	return Vector3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(Vector3 &a, Real b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ Vector3i operator-(Vector3i a, Vector3i b)
{
	return Vector3i(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(Vector3i &a, Vector3i b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ Vector3i operator-(Vector3i a, Int b)
{
	return Vector3i(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ Vector3i operator-(Int b, Vector3i a)
{
	return Vector3i(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(Vector3i &a, Int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ Vector3ui operator-(Vector3ui a, Vector3ui b)
{
	return Vector3ui(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(Vector3ui &a, Vector3ui b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ Vector3ui operator-(Vector3ui a, UInt b)
{
	return Vector3ui(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ Vector3ui operator-(UInt b, Vector3ui a)
{
	return Vector3ui(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(Vector3ui &a, UInt b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ Vector4 operator-(Vector4 a, Vector4 b)
{
	return Vector4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(Vector4 &a, Vector4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ Vector4 operator-(Vector4 a, Real b)
{
	return Vector4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(Vector4 &a, Real b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ Vector4i operator-(Vector4i a, Vector4i b)
{
	return Vector4i(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(Vector4i &a, Vector4i b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ Vector4i operator-(Vector4i a, Int b)
{
	return Vector4i(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ Vector4i operator-(Int b, Vector4i a)
{
	return Vector4i(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(Vector4i &a, Int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ Vector4ui operator-(Vector4ui a, Vector4ui b)
{
	return Vector4ui(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(Vector4ui &a, Vector4ui b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ Vector4ui operator-(Vector4ui a, UInt b)
{
	return Vector4ui(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ Vector4ui operator-(UInt b, Vector4ui a)
{
	return Vector4ui(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(Vector4ui &a, UInt b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 operator*(Vector2 a, Vector2 b)
{
	return Vector2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(Vector2 &a, Vector2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ Vector2 operator*(Vector2 a, Real b)
{
	return Vector2(a.x * b, a.y * b);
}
inline __host__ __device__ Vector2 operator*(Real b, Vector2 a)
{
	return Vector2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(Vector2 &a, Real b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ Vector2i operator*(Vector2i a, Vector2i b)
{
	return Vector2i(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(Vector2i &a, Vector2i b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ Vector2i operator*(Vector2i a, Int b)
{
	return Vector2i(a.x * b, a.y * b);
}
inline __host__ __device__ Vector2i operator*(Int b, Vector2i a)
{
	return Vector2i(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(Vector2i &a, Int b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ Vector2ui operator*(Vector2ui a, Vector2ui b)
{
	return Vector2ui(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(Vector2ui &a, Vector2ui b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ Vector2ui operator*(Vector2ui a, UInt b)
{
	return Vector2ui(a.x * b, a.y * b);
}
inline __host__ __device__ Vector2ui operator*(UInt b, Vector2ui a)
{
	return Vector2ui(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(Vector2ui &a, UInt b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ Vector3 operator*(Vector3 a, Vector3 b)
{
	return Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(Vector3 &a, Vector3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ Vector3 operator*(Vector3 a, Real b)
{
	return Vector3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ Vector3 operator*(Real b, Vector3 a)
{
	return Vector3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(Vector3 &a, Real b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ Vector3i operator*(Vector3i a, Vector3i b)
{
	return Vector3i(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(Vector3i &a, Vector3i b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ Vector3i operator*(Vector3i a, Int b)
{
	return Vector3i(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ Vector3i operator*(Int b, Vector3i a)
{
	return Vector3i(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(Vector3i &a, Int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ Vector3ui operator*(Vector3ui a, Vector3ui b)
{
	return Vector3ui(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(Vector3ui &a, Vector3ui b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ Vector3ui operator*(Vector3ui a, UInt b)
{
	return Vector3ui(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ Vector3ui operator*(UInt b, Vector3ui a)
{
	return Vector3ui(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(Vector3ui &a, UInt b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ Vector4 operator*(Vector4 a, Vector4 b)
{
	return Vector4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(Vector4 &a, Vector4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ Vector4 operator*(Vector4 a, Real b)
{
	return Vector4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ Vector4 operator*(Real b, Vector4 a)
{
	return Vector4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(Vector4 &a, Real b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ Vector4i operator*(Vector4i a, Vector4i b)
{
	return Vector4i(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(Vector4i &a, Vector4i b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ Vector4i operator*(Vector4i a, Int b)
{
	return Vector4i(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ Vector4i operator*(Int b, Vector4i a)
{
	return Vector4i(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(Vector4i &a, Int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ Vector4ui operator*(Vector4ui a, Vector4ui b)
{
	return Vector4ui(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(Vector4ui &a, Vector4ui b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ Vector4ui operator*(Vector4ui a, UInt b)
{
	return Vector4ui(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ Vector4ui operator*(UInt b, Vector4ui a)
{
	return Vector4ui(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(Vector4ui &a, UInt b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 operator/(Vector2 a, Vector2 b)
{
	return Vector2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(Vector2 &a, Vector2 b)
{
	a.x /= b.x;
	a.y /= b.y;
}
inline __host__ __device__ Vector2 operator/(Vector2 a, Real b)
{
	return Vector2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(Vector2 &a, Real b)
{
	a.x /= b;
	a.y /= b;
}
inline __host__ __device__ Vector2 operator/(Real b, Vector2 a)
{
	return Vector2(b / a.x, b / a.y);
}

inline __host__ __device__ Vector3 operator/(Vector3 a, Vector3 b)
{
	return Vector3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(Vector3 &a, Vector3 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
inline __host__ __device__ Vector3 operator/(Vector3 a, Real b)
{
	return Vector3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(Vector3 &a, Real b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
inline __host__ __device__ Vector3 operator/(Real b, Vector3 a)
{
	return Vector3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ Vector4 operator/(Vector4 a, Vector4 b)
{
	return Vector4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(Vector4 &a, Vector4 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}
inline __host__ __device__ Vector4 operator/(Vector4 a, Real b)
{
	return Vector4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(Vector4 &a, Real b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}
inline __host__ __device__ Vector4 operator/(Real b, Vector4 a)
{
	return Vector4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// minM
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ Vector2 fminf(Vector2 a, Vector2 b)
{
	return Vector2(fminfM(a.x, b.x), fminfM(a.y, b.y));
}
inline __host__ __device__ Vector3 fminf(Vector3 a, Vector3 b)
{
	return Vector3(fminfM(a.x, b.x), fminfM(a.y, b.y), fminfM(a.z, b.z));
}
inline  __host__ __device__ Vector4 fminf(Vector4 a, Vector4 b)
{
	return Vector4(fminfM(a.x, b.x), fminfM(a.y, b.y), fminfM(a.z, b.z), fminfM(a.w, b.w));
}

inline __host__ __device__ Vector2i minM(Vector2i a, Vector2i b)
{
	return Vector2i(minM(a.x, b.x), minM(a.y, b.y));
}
inline __host__ __device__ Vector3i minM(Vector3i a, Vector3i b)
{
	return Vector3i(minM(a.x, b.x), minM(a.y, b.y), minM(a.z, b.z));
}
inline __host__ __device__ Vector4i minM(Vector4i a, Vector4i b)
{
	return Vector4i(minM(a.x, b.x), minM(a.y, b.y), minM(a.z, b.z), minM(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// maxM
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 fmaxf(Vector2 a, Vector2 b)
{
	return Vector2(fmaxfM(a.x, b.x), fmaxfM(a.y, b.y));
}
inline __host__ __device__ Vector3 fmaxf(Vector3 a, Vector3 b)
{
	return Vector3(fmaxfM(a.x, b.x), fmaxfM(a.y, b.y), fmaxfM(a.z, b.z));
}
inline __host__ __device__ Vector4 fmaxf(Vector4 a, Vector4 b)
{
	return Vector4(fmaxfM(a.x, b.x), fmaxfM(a.y, b.y), fmaxfM(a.z, b.z), fmaxfM(a.w, b.w));
}

inline __host__ __device__ Vector2i max(Vector2i a, Vector2i b)
{
	return Vector2i(maxM(a.x, b.x), maxM(a.y, b.y));
}
inline __host__ __device__ Vector3i max(Vector3i a, Vector3i b)
{
	return Vector3i(maxM(a.x, b.x), maxM(a.y, b.y), maxM(a.z, b.z));
}
inline __host__ __device__ Vector4i max(Vector4i a, Vector4i b)
{
	return Vector4i(maxM(a.x, b.x), maxM(a.y, b.y), maxM(a.z, b.z), maxM(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ Real lerp(Real a, Real b, Real t)
{
	return a + t * (b - a);
}
inline __device__ __host__ Vector2 lerp(Vector2 a, Vector2 b, Real t)
{
	return a + t * (b - a);
}
inline __device__ __host__ Vector3 lerp(Vector3 a, Vector3 b, Real t)
{
	return a + t * (b - a);
}
inline __device__ __host__ Vector4 lerp(Vector4 a, Vector4 b, Real t)
{
	return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// clampM
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ Real clampM(Real f, Real a, Real b)
{
	return fmaxfM(a, fminfM(f, b));
}
inline __device__ __host__ Int clampM(Int f, Int a, Int b)
{
	return maxM(a, minM(f, b));
}
inline __device__ __host__ Vector2 clampM(Vector2 v, Real a, Real b)
{
	return Vector2(clampM(v.x, a, b), clampM(v.y, a, b));
}
inline __device__ __host__ Vector2 clampM(Vector2 v, Vector2 a, Vector2 b)
{
	return Vector2(clampM(v.x, a.x, b.x), clampM(v.y, a.y, b.y));
}
inline __device__ __host__ Vector3 clampM(Vector3 v, Real a, Real b)
{
	return Vector3(clampM(v.x, a, b), clampM(v.y, a, b), clampM(v.z, a, b));
}
inline __device__ __host__ Vector3 clampM(Vector3 v, Vector3 a, Vector3 b)
{
	return Vector3(clampM(v.x, a.x, b.x), clampM(v.y, a.y, b.y), clampM(v.z, a.z, b.z));
}
inline __device__ __host__ Vector4 clampM(Vector4 v, Real a, Real b)
{
	return Vector4(clampM(v.x, a, b), clampM(v.y, a, b), clampM(v.z, a, b), clampM(v.w, a, b));
}
inline __device__ __host__ Vector4 clampM(Vector4 v, Vector4 a, Vector4 b)
{
	return Vector4(clampM(v.x, a.x, b.x), clampM(v.y, a.y, b.y), clampM(v.z, a.z, b.z), clampM(v.w, a.w, b.w));
}

inline __device__ __host__ Vector2i clampM(Vector2i v, Int a, Int b)
{
	return Vector2i(clampM(v.x, a, b), clampM(v.y, a, b));
}
inline __device__ __host__ Vector2i clampM(Vector2i v, Vector2i a, Vector2i b)
{
	return Vector2i(clampM(v.x, a.x, b.x), clampM(v.y, a.y, b.y));
}
inline __device__ __host__ Vector3i clampM(Vector3i v, Int a, Int b)
{
	return Vector3i(clampM(v.x, a, b), clampM(v.y, a, b), clampM(v.z, a, b));
}
inline __device__ __host__ Vector3i clampM(Vector3i v, Vector3i a, Vector3i b)
{
	return Vector3i(clampM(v.x, a.x, b.x), clampM(v.y, a.y, b.y), clampM(v.z, a.z, b.z));
}
inline __device__ __host__ Vector4i clampM(Vector4i v, Int a, Int b)
{
	return Vector4i(clampM(v.x, a, b), clampM(v.y, a, b), clampM(v.z, a, b), clampM(v.w, a, b));
}
inline __device__ __host__ Vector4i clampM(Vector4i v, Vector4i a, Vector4i b)
{
	return Vector4i(clampM(v.x, a.x, b.x), clampM(v.y, a.y, b.y), clampM(v.z, a.z, b.z), clampM(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Real dot(Vector2 a, Vector2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ Real dot(Vector3 a, Vector3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ Real dot(Vector4 a, Vector4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ Int dot(Vector2i a, Vector2i b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ Int dot(Vector3i a, Vector3i b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ Int dot(Vector4i a, Vector4i b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ UInt dot(Vector2ui a, Vector2ui b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ UInt dot(Vector3ui a, Vector3ui b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ UInt dot(Vector4ui a, Vector4ui b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Real length(Vector2 v)
{
	return sqrt(dot(v, v));
}
inline __host__ __device__ Real length(Vector3 v)
{
	return sqrt(dot(v, v));
}
inline __host__ __device__ Real length(Vector4 v)
{
	return sqrt(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

//inline __host__ __device__ Vector2 normalize(Vector2 v)
//{
//	Real invLen = rsqrtM(dot(v, v));
//	return v * invLen;
//}
//inline __host__ __device__ Vector3 normalize(Vector3 v)
//{
//	Real invLen = rsqrtM(dot(v, v));
//	return v * invLen;
//}
//inline __host__ __device__ Vector4 normalize(Vector4 v)
//{
//	Real invLen = rsqrtM(dot(v, v));
//	return v * invLen;
//}

////////////////////////////////////////////////////////////////////////////////
// floorM
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 floorM(Vector2 v)
{
	return Vector2(floor(v.x), floor(v.y));
}
inline __host__ __device__ Vector3 floorM(Vector3 v)
{
	return Vector3(floor(v.x), floor(v.y), floor(v.z));
}
inline __host__ __device__ Vector4 floorM(Vector4 v)
{
	return Vector4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Real fracf(Real v)
{
	return v - floor(v);
}
inline __host__ __device__ Vector2 fracf(Vector2 v)
{
	return Vector2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ Vector3 fracf(Vector3 v)
{
	return Vector3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ Vector4 fracf(Vector4 v)
{
	return Vector4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 fmod(Vector2 a, Vector2 b)
{
	return Vector2(fmod(a.x, b.x), fmod(a.y, b.y));
}
inline __host__ __device__ Vector3 fmod(Vector3 a, Vector3 b)
{
	return Vector3(fmod(a.x, b.x), fmod(a.y, b.y), fmod(a.z, b.z));
}
inline __host__ __device__ Vector4 fmod(Vector4 a, Vector4 b)
{
	return Vector4(fmod(a.x, b.x), fmod(a.y, b.y), fmod(a.z, b.z), fmod(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector2 fabs(Vector2 v)
{
	return Vector2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ Vector3 fabs(Vector3 v)
{
	return Vector3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ Vector4 fabs(Vector4 v)
{
	return Vector4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ Vector2i abs(Vector2i v)
{
	return Vector2i(abs(v.x), abs(v.y));
}
inline __host__ __device__ Vector3i abs(Vector3i v)
{
	return Vector3i(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ Vector4i abs(Vector4i v)
{
	return Vector4i(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector3 reflect(Vector3 i, Vector3 n)
{
	return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ Vector3 cross(Vector3 a, Vector3 b)
{
	return Vector3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ Real smoothstep(Real a, Real b, Real x)
{
	Real y = clampM((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ Vector2 smoothstep(Vector2 a, Vector2 b, Vector2 x)
{
	Vector2 y = clampM((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(Vector2(3.0f, 3.0f) - (Vector2(2.0f, 2.0f)*y)));
}
inline __device__ __host__ Vector3 smoothstep(Vector3 a, Vector3 b, Vector3 x)
{
	Vector3 y = clampM((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(Vector3(3.0f, 3.0f, 3.0f) - (Vector3(2.0f, 2.0f, 2.0f)*y)));
}
inline __device__ __host__ Vector4 smoothstep(Vector4 a, Vector4 b, Vector4 x)
{
	Vector4 y = clampM((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(Vector4(3.0f, 3.0f, 3.0f, 3.0f) - (Vector4(2.0f, 2.0f, 2.0f, 2.0f)*y)));
}


inline __host__ __device__ bool operator==(const Vector3i& vA, const Vector3i& vB)
{
	return vA.x == vB.x && vA.y == vB.y && vA.z == vB.z;
}

inline __host__ __device__ bool operator==(const Vector3& vA, const Vector3& vB)
{
	return vA.x == vB.x && vA.y == vB.y && vA.z == vB.z;
}

inline __host__ __device__ Vector3ui castToVector3ui(const Vector3i& vInput)
{
	return Vector3ui(static_cast<UInt>(vInput.x), static_cast<UInt>(vInput.y), static_cast<UInt>(vInput.z));
}

inline __host__ __device__  Vector3ui castToVector3ui(const Vector3& vInput)
{
	return Vector3ui(static_cast<UInt>(vInput.x), static_cast<UInt>(vInput.y), static_cast<UInt>(vInput.z));
}

inline __host__ __device__  Vector3i castToVector3i(const Vector3& vInput)
{
	return Vector3i(static_cast<Int>(vInput.x), static_cast<Int>(vInput.y), static_cast<Int>(vInput.z));
}

inline  __host__ __device__ Vector3i castToVector3i(const Vector3ui& vInput)
{
	return Vector3i(static_cast<Int>(vInput.x), static_cast<Int>(vInput.y), static_cast<Int>(vInput.z));
}

inline __host__ __device__  Vector3i ceilToVector3i(const Vector3& vInput)
{
	return Vector3i(static_cast<Int>(ceil(vInput.x)), static_cast<Int>(ceil(vInput.y)), static_cast<Int>(ceil(vInput.z)));
}

inline __host__ __device__  Vector3ui ceilToVector3ui(const Vector3& vInput)
{
	return Vector3ui(static_cast<UInt>(ceil(vInput.x)), static_cast<UInt>(ceil(vInput.y)), static_cast<UInt>(ceil(vInput.z)));
}

inline __host__ __device__  Vector3 castToVector3(const Vector3i& vInput)
{
	return Vector3(static_cast<Real>(vInput.x), static_cast<Real>(vInput.y), static_cast<Real>(vInput.z));
}

inline  __host__ __device__ Vector3 castToVector3(const Vector3ui& vInput)
{
	return Vector3(static_cast<Real>(vInput.x), static_cast<Real>(vInput.y), static_cast<Real>(vInput.z));
}

inline  __host__ __device__ Vector3 round(const Vector3& vInput)
{
	return Vector3(round(vInput.x), round(vInput.y), round(vInput.z));
}

inline  __host__ __device__ Vector3 normalize(const Vector3& vInput)
{
	return vInput / length(vInput);
}

inline  __host__ __device__ Real get(const Vector3& vInput, UInt vIndex)
{
	switch (vIndex)
	{
	case 0:
		return vInput.x;
	case 1:
		return vInput.y;
	case 2:
		return vInput.z;
	default:
		return vInput.x;
		break;
	}
}

struct SMatrix3x3
{
	Vector3 row0;
	Vector3 row1;
	Vector3 row2;

	__host__ __device__ SMatrix3x3()
	{
		row0 = Vector3(1, 0, 0);
		row1 = Vector3(0, 1, 0);
		row2 = Vector3(0, 0, 1);
	}

	__host__ __device__ Real operator()(Int vRow, Int vCol)
	{
		switch (vRow)
		{
		case 0:
			return get(row0, vCol);
		case 1:
			return get(row1, vCol);
		case 2:
			return get(row2, vCol);
		default:
			return get(row0, vCol);
			break;
		}
	}

	__host__ __device__ SMatrix3x3 getInv()
	{
		double det = (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(2, 1) * (*this)(1, 2)) -
			(*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
			(*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));

		double invdet = 1 / det;

		SMatrix3x3 Inv; // inverse of matrix (*this)
		Inv.row0 = Vector3
		(
			((*this)(1, 1) * (*this)(2, 2) - (*this)(2, 1) * (*this)(1, 2)) * invdet,
			((*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2)) * invdet,
			((*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1)) * invdet
		);

		Inv.row1 = Vector3
		(
			((*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2)) * invdet,
		    ((*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0)) * invdet,
			((*this)(1, 0) * (*this)(0, 2) - (*this)(0, 0) * (*this)(1, 2)) * invdet
		);

		Inv.row2 = Vector3
		(
			((*this)(1, 0) * (*this)(2, 1) - (*this)(2, 0) * (*this)(1, 1)) * invdet,
			((*this)(2, 0) * (*this)(0, 1) - (*this)(0, 0) * (*this)(2, 1)) * invdet,
			((*this)(0, 0) * (*this)(1, 1) - (*this)(1, 0) * (*this)(0, 1)) * invdet
		);

		return Inv;
	}

	__host__ __device__ void setIdentity()
	{
		row0.x = 1.0;
		row1.y = 1.0;
		row2.z = 1.0;
	}

	__host__ __device__ void setZero()
	{
		row0 = Vector3(0, 0, 0);
		row1 = Vector3(0, 0, 0);
		row2 = Vector3(0, 0, 0);
	}

	__host__ __device__ void operator =(const SMatrix3x3& vOperand)
	{
		row0 = vOperand.row0;
		row1 = vOperand.row1;
		row2 = vOperand.row2;
	}

	__host__ __device__ void operator +=(const SMatrix3x3& vOperand)
	{
		row0 += vOperand.row0;
		row1 += vOperand.row1;
		row2 += vOperand.row2;
	}

	__host__ __device__ SMatrix3x3 transpose() const
	{
		SMatrix3x3 Result;

		Result.row0.x = row0.x;
		Result.row0.y = row1.x;
		Result.row0.z = row2.x;
		Result.row1.x = row0.y;
		Result.row1.y = row1.y;
		Result.row1.z = row2.y;
		Result.row2.x = row0.z;
		Result.row2.y = row1.z;
		Result.row2.z = row2.z;

		return Result;
	}

	__host__ __device__ Real squaredFNorm() const
	{
		return row0.x*row0.x + row0.y*row0.y + row0.z*row0.z +
			row1.x*row1.x + row1.y*row1.y + row1.z*row1.z +
			row2.x*row1.x + row2.y*row2.y + row2.z*row2.z;
	}

	__host__ __device__ Real trace() const
	{
		return row0.x + row1.y + row2.z;
	}

	__host__ __device__ Vector3 col(int vIndex) const
	{
		Vector3 Result;

		switch (vIndex)
		{
		case 0:
			Result.x = row0.x;
			Result.y = row1.x;
			Result.z = row2.x;
			break;
		case 1:
			Result.x = row0.y;
			Result.y = row1.y;
			Result.z = row2.y;
			break;
		case 2:
			Result.x = row0.z;
			Result.y = row1.z;
			Result.z = row2.z;
			break;
		default:
			break;
		}

		return Result;
	}

	__host__ __device__ void setCol(int vIndex, Vector3 vVector)
	{
		switch (vIndex)
		{
		case 0:
			row0.x = vVector.x;
			row1.x = vVector.y;
			row2.x = vVector.z;
			break;
		case 1:
			row0.y = vVector.x;
			row1.y = vVector.y;
			row2.y = vVector.z;
			break;
		case 2:
			row0.z = vVector.x;
			row1.z = vVector.y;
			row2.z = vVector.z;
			break;
		default:
			break;
		}
	}

	__host__ __device__ SMatrix3x3 operator -(const SMatrix3x3& vOperand) const
	{
		SMatrix3x3 Result;

		Result.row0 = row0 - vOperand.row0;
		Result.row1 = row1 - vOperand.row1;
		Result.row2 = row2 - vOperand.row2;

		return Result;
	}

	__host__ __device__ SMatrix3x3 operator +(const SMatrix3x3& vOperand) const
	{
		SMatrix3x3 Result;

		Result.row0 = row0 + vOperand.row0;
		Result.row1 = row1 + vOperand.row1;
		Result.row2 = row2 + vOperand.row2;

		return Result;
	}

	__host__ __device__ SMatrix3x3 operator *(const SMatrix3x3& vOperand) const
	{
		SMatrix3x3 Result;

		Result.row0.x = row0.x*vOperand.row0.x + row0.y*vOperand.row1.x + row0.z*vOperand.row2.x;
		Result.row0.y = row0.x*vOperand.row0.y + row0.y*vOperand.row1.y + row0.z*vOperand.row2.y;
		Result.row0.z = row0.x*vOperand.row0.z + row0.y*vOperand.row1.z + row0.z*vOperand.row2.z;
		Result.row1.x = row1.x*vOperand.row0.x + row1.y*vOperand.row1.x + row1.z*vOperand.row2.x;
		Result.row1.y = row1.x*vOperand.row0.y + row1.y*vOperand.row1.y + row1.z*vOperand.row2.y;
		Result.row1.z = row1.x*vOperand.row0.z + row1.y*vOperand.row1.z + row1.z*vOperand.row2.z;
		Result.row2.x = row2.x*vOperand.row0.x + row2.y*vOperand.row1.x + row2.z*vOperand.row2.x;
		Result.row2.y = row2.x*vOperand.row0.y + row2.y*vOperand.row1.y + row2.z*vOperand.row2.y;
		Result.row2.z = row2.x*vOperand.row0.z + row2.y*vOperand.row1.z + row2.z*vOperand.row2.z;

		return Result;
	}

	__host__ __device__ Vector3 operator *(const Vector3& vOperand) const
	{
		Vector3 Result;

		Result.x = row0.x*vOperand.x + row0.y*vOperand.y + row0.z*vOperand.z;
		Result.y = row1.x*vOperand.x + row1.y*vOperand.y + row1.z*vOperand.z;
		Result.z = row2.x*vOperand.x + row2.y*vOperand.y + row2.z*vOperand.z;

		return Result;
	}
};