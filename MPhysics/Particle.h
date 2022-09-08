#pragma once
#include "Common.h"
#include "CuDenseVector.h"

enum EInterpolationMethod
{
	EXPLICIT_EULER,
	IMPLICIT_EULER
};

#define ParticlePos(vPosVector, vIndex)							\
	Vector3														\
	(															\
		ParticlePosPortion(vPosVector, vIndex, 0),				\
		ParticlePosPortion(vPosVector, vIndex, 1),				\
		ParticlePosPortion(vPosVector, vIndex, 2)				\
	)

#define ParticlePosPortion(vPosVector, vIndex, vDim) (vPosVector)[(vIndex) * 3 + (vDim)]

class CParticleGroup
{
public:
	CParticleGroup() = default;
	~CParticleGroup();

	void generateRandomData(SAABB vSpaceRange, Real vParticleRadius);
	void freshParticlePos(EInterpolationMethod vVelInteType, Real vDeltaT);
	void freshParticleVel(EInterpolationMethod vVelInteType, Real vDeltaT);
	void addAccelerationToVel(const CCuDenseVector& vAcceleration, Real vDeltaT);

	void appendParticleBlock(SAABB vSpaceRange);
	void appendParticles(const vector<Real>& vNewParticlesPosCPU);
	void appendEmptyParticle(UInt vSize);
	void reduceEmptyParticle();
	void removeParticles(const thrust::device_vector<bool>& vFilterMap);

	void setParticleRadius(Real vInput);
	Real getParticleRadius() const;
	Real getParticleSupportRadius() const;

	UInt getSize() const;
	UInt getRealSize() const;
	Real getParticleVolume() const;
	Vector3 getParticlePos(UInt vIndex) const;
	Vector3 getParticleVel(UInt vIndex) const;
	Vector3 getPrevParticlePos(UInt vIndex) const;

	void setParticlePos(const CCuDenseVector & vNewPos);
	void setParticleVel (const CCuDenseVector & vNewVel);
	void setPrevParticlePos(const CCuDenseVector & vNewPos);

	const CCuDenseVector & getConstParticlePos()const;
	const CCuDenseVector & getConstParticleVel()const;
	const CCuDenseVector & getConstPrevParticlePos()const;

	CCuDenseVector & getParticlePos();
	CCuDenseVector & getParticleVel();
	CCuDenseVector & getPrevParticlePos();
	thrust::device_vector<Real> & getLiveTime();

	Real* getParticlePosGPUPtr();
	Real* getPrevParticlePosGPUPtr();
	Real* getParticleVelGPUPtr();
	Real* getParticleLiveTimeGPUPtr();
	const Real* getConstParticlePosGPUPtr() const;
	const Real* getConstParticleVelGPUPtr() const;

protected:
	UInt m_Size = 0;
	UInt m_RealSize = 0;
	Real m_ParticleRadius = static_cast<Real>(0.025);
	Real m_ParticleSupportRadius = static_cast<Real>(0.1);
	Real m_ParticleVolume = static_cast<Real>(0.0001);

	CCuDenseVector m_ParticlePos;
	CCuDenseVector m_PrevParticlePos;
	CCuDenseVector m_ParticleVel;

	thrust::device_vector<Real> m_LiveTime;//粒子的删除定时器
};