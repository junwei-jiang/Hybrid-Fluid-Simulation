#include "SPHKernelFunc.cuh"

__host__ __device__ void CCubicKernel::setRadius(Real val)
{
	m_radius = val;
	const Real pi = static_cast<Real>(M_PI);

	const Real h3 = m_radius * m_radius*m_radius;
	m_k = static_cast<Real>(8.0) / (pi*h3);
	m_l = static_cast<Real>(48.0) / (pi*h3);
	m_W_zero = W(Vector3(0, 0, 0));
}

__host__ __device__  Real  CCubicKernel::W(const Real r)
{
	Real res = 0.0;
	const Real q = r / m_radius;
	if (q <= 1.0)
	{
		if (q <= 0.5)
		{
			const Real q2 = q * q;
			const Real q3 = q2 * q;
			res = m_k * (static_cast<Real>(6.0)*q3 - static_cast<Real>(6.0)*q2 + static_cast<Real>(1.0));
		}
		else
		{
			res = m_k * (static_cast<Real>(2.0)*pow(static_cast<Real>(1.0) - q, static_cast<Real>(3.0)));
		}
	}
	return res;
}

__host__ __device__  Real CCubicKernel::W(const Vector3 &r)
{
	return W(length(r));
}

__host__ __device__  Vector3 CCubicKernel::gradW(const Vector3 &r)
{
	Vector3 res;
	const Real rl = length(r);
	const Real q = rl / m_radius;
	if ((rl > 1.0e-5) && (q <= 1.0))
	{
		const Vector3 gradq = r * (static_cast<Real>(1.0) / (rl*m_radius));
		if (q <= 0.5)
		{
			res = m_l * q*((Real) 3.0*q - static_cast<Real>(2.0))*gradq;
		}
		else
		{
			const Real factor = static_cast<Real>(1.0) - q;
			res = m_l * (-factor * factor)*gradq;
		}
	}
	else
		res = Vector3(0, 0, 0);

	return res;
}

__host__ __device__  Real CCubicKernel::W_zero()
{
	return m_W_zero;
}

__constant__ __device__ CCubicKernel SmoothKernelCubic;