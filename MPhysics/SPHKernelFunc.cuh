#pragma once
#include "Common.h"

class CCubicKernel
{
public:
	__host__ __device__  Real getRadius() { return m_radius; }
	__host__ __device__  Real getK() { return m_k; }
	__host__ __device__  Real getL() { return m_l; }
	__host__ __device__  void setRadius(Real val);

	__host__ __device__  Real W(const Real r);

	__host__ __device__  Real W(const Vector3 &r);

	__host__ __device__  Vector3 gradW(const Vector3 &r);

	__host__ __device__  Real W_zero();

private:
	Real m_radius;
	Real m_k;
	Real m_l;
	Real m_W_zero;
};

__constant__ __device__ extern CCubicKernel SmoothKernelCubic;