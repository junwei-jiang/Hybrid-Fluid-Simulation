#include "Particle.h"
#include "ThrustWapper.cuh"
#include "SimulationConfigManager.h"
#include "ParticleKernel.cuh"
#include <vector>
#include <cstdlib>
#include <ctime>

CParticleGroup::~CParticleGroup(){}

void CParticleGroup::generateRandomData(SAABB vSpaceRange, Real vParticleRadius)
{
	setParticleRadius(vParticleRadius);

	Vector3 RangeLength = vSpaceRange.Max - vSpaceRange.Min;
	Vector3ui Dim = castToVector3ui(round(RangeLength / (vParticleRadius * 2.0))) - 1;
	UInt Size = Dim.x * Dim.y * Dim.z;
	vector<Real> PosCPU(Size * 3);

	Vector3 RangeSize = vSpaceRange.Max - vSpaceRange.Min;
	srand((Int)time(0));
	for (UInt i = 0; i < Size; i++)
	{
		Vector3 Point;
		Point.x = ((Real)rand() / (Real)RAND_MAX) * RangeSize.x;
		Point.y = ((Real)rand() / (Real)RAND_MAX) * RangeSize.y;
		Point.z = ((Real)rand() / (Real)RAND_MAX) * RangeSize.z;
		Point += vSpaceRange.Min;
		ParticlePosPortion(PosCPU, i, 0) = Point.x;
		ParticlePosPortion(PosCPU, i, 1) = Point.y;
		ParticlePosPortion(PosCPU, i, 2) = Point.z;
	}

	appendParticles(PosCPU);
}

void CParticleGroup::appendParticleBlock(SAABB vSpaceRange)
{
	Vector3 RangeLength = vSpaceRange.Max - vSpaceRange.Min;
	Vector3ui Dim = castToVector3ui(round(RangeLength / (m_ParticleRadius * 2.0))) - 1;
	UInt Size = Dim.x * Dim.y * Dim.z;

	vector<Real> PosCPU(Size * 3);
	UInt Count = 0;
	for (int i = 0; i < Dim.x; i++)
	{
		for (int k = 0; k < Dim.y; k++)
		{
			for (int j = 0; j < Dim.z; j++)
			{
				ParticlePosPortion(PosCPU, Count, 0) = vSpaceRange.Min.x + m_ParticleRadius * 2.0 + i * m_ParticleRadius * 2.0;
				ParticlePosPortion(PosCPU, Count, 1) = vSpaceRange.Min.y + m_ParticleRadius * 2.0 + k * m_ParticleRadius * 2.0;
				ParticlePosPortion(PosCPU, Count, 2) = vSpaceRange.Min.z + m_ParticleRadius * 2.0 + j * m_ParticleRadius * 2.0;
				Count++;
			}
		}
	}

	appendParticles(PosCPU);
}

void CParticleGroup::appendParticles(const vector<Real>& vNewParticlesPosCPU)
{
	_ASSERT(vNewParticlesPosCPU.size() % 3 == 0);

	m_Size += (vNewParticlesPosCPU.size() / 3);

	m_ParticlePos.append(vNewParticlesPosCPU);
	m_PrevParticlePos.append(vNewParticlesPosCPU);
	m_ParticleVel.resize(m_Size * 3, 0);
	resizeDeviceVector(m_LiveTime, m_Size * 3, 0);

	m_RealSize = m_Size;
}

void CParticleGroup::freshParticlePos(EInterpolationMethod vVelInteType, Real vDeltaT)
{
	switch (vVelInteType)
	{
	case EXPLICIT_EULER:
		m_PrevParticlePos = m_ParticlePos;
		m_ParticlePos.plusAlphaX(m_ParticleVel, vDeltaT);
		break;
	}
}

void CParticleGroup::freshParticleVel(EInterpolationMethod vVelInteType, Real vDeltaT)
{
	switch (vVelInteType)
	{
	case EXPLICIT_EULER:
		m_ParticleVel = m_ParticlePos;
		m_ParticleVel.plusAlphaX(m_PrevParticlePos, -1.0);
		m_ParticleVel.scale(1.0 / vDeltaT);
		break;
	}
}

void CParticleGroup::addAccelerationToVel(const CCuDenseVector & vAcceleration, Real vDeltaT)
{
	m_ParticleVel.plusAlphaX(vAcceleration, vDeltaT);
}

void CParticleGroup::appendEmptyParticle(UInt vSize)
{
	appendEmptyParticleInvoker
	(
		m_ParticlePos.getVectorValue(),
		m_PrevParticlePos.getVectorValue(),
		m_ParticleVel.getVectorValue(),
		m_LiveTime,
		m_RealSize,
		vSize,
		m_Size
	);
}

void CParticleGroup::reduceEmptyParticle()
{
	reduceEmptyParticleInvoker
	(
		m_ParticlePos.getVectorValue(),
		m_PrevParticlePos.getVectorValue(),
		m_ParticleVel.getVectorValue(),
		m_LiveTime,
		m_RealSize,
		m_Size
	);
	m_ParticlePos.updateSize(m_Size * 3);
	m_PrevParticlePos.updateSize(m_Size * 3);
	m_ParticleVel.updateSize(m_Size * 3);
}

void CParticleGroup::removeParticles(const thrust::device_vector<bool>& vFilterMap)
{
	reduceParticleInvoker
	(
		vFilterMap,
		m_ParticlePos.getVectorValue(),
		m_PrevParticlePos.getVectorValue(),
		m_ParticleVel.getVectorValue(),
		m_LiveTime,
		m_Size
	);
	m_ParticlePos.updateSize(m_Size);
	m_PrevParticlePos.updateSize(m_Size);
	m_ParticleVel.updateSize(m_Size);
}

void CParticleGroup::setParticleRadius(Real vInput)
{
	m_ParticleRadius = vInput;
	const Real Diam = static_cast<Real>(2.0) * m_ParticleRadius;
	m_ParticleVolume = static_cast<Real>(0.8) * Diam * Diam * Diam;
	m_ParticleSupportRadius = static_cast<Real>(4.0) * m_ParticleRadius;
}

Real CParticleGroup::getParticleRadius() const
{
	return m_ParticleRadius;
}

Real CParticleGroup::getParticleSupportRadius() const
{
	return m_ParticleSupportRadius;
}

UInt CParticleGroup::getSize() const
{
	return m_Size;
}

UInt CParticleGroup::getRealSize() const
{
	return m_RealSize;
}

Real CParticleGroup::getParticleVolume() const
{
	return m_ParticleVolume;
}

const Real* CParticleGroup::getConstParticlePosGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ParticlePos.getConstVectorValue());
}

const Real* CParticleGroup::getConstParticleVelGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ParticleVel.getConstVectorValue());
}

Vector3 CParticleGroup::getParticlePos(UInt vIndex) const
{
	_ASSERT(vIndex < m_Size);
	return ParticlePos(m_ParticlePos, vIndex);
}

CCuDenseVector & CParticleGroup::getParticlePos()
{
	return m_ParticlePos;
}

CCuDenseVector & CParticleGroup::getParticleVel()
{
	return m_ParticleVel;
}

Vector3 CParticleGroup::getParticleVel(UInt vIndex) const
{
	_ASSERT(vIndex < m_Size);
	return ParticlePos(m_ParticleVel, vIndex);
}

Vector3 CParticleGroup::getPrevParticlePos(UInt vIndex) const
{
	_ASSERT(vIndex < m_Size);
	return ParticlePos(m_PrevParticlePos, vIndex);
}

CCuDenseVector & CParticleGroup::getPrevParticlePos()
{
	return m_PrevParticlePos;
}

thrust::device_vector<Real>& CParticleGroup::getLiveTime()
{
	return m_LiveTime;
}

void CParticleGroup::setParticlePos(const CCuDenseVector & vNewPos)
{
	m_ParticlePos = vNewPos;
}

void CParticleGroup::setParticleVel(const CCuDenseVector & vNewVel)
{
	m_ParticleVel = vNewVel;
}

void CParticleGroup::setPrevParticlePos(const CCuDenseVector & vNewPos)
{
	m_PrevParticlePos = vNewPos;
}

const CCuDenseVector & CParticleGroup::getConstParticlePos() const
{
	return m_ParticlePos;
}

const CCuDenseVector & CParticleGroup::getConstParticleVel() const
{
	return m_ParticleVel;
}

const CCuDenseVector & CParticleGroup::getConstPrevParticlePos() const
{
	return m_PrevParticlePos;
}

Real * CParticleGroup::getParticlePosGPUPtr()
{
	return getRawDevicePointerReal(m_ParticlePos.getVectorValue());
}

Real* CParticleGroup::getPrevParticlePosGPUPtr()
{
	return getRawDevicePointerReal(m_PrevParticlePos.getVectorValue());
}

Real * CParticleGroup::getParticleVelGPUPtr()
{
	return getRawDevicePointerReal(m_ParticleVel.getVectorValue());
}

Real * CParticleGroup::getParticleLiveTimeGPUPtr()
{
	return getRawDevicePointerReal(m_LiveTime);
}
