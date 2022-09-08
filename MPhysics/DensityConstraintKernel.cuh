#pragma once
#include "Common.h"
#include "CuDenseVector.h"
#include "BoundaryVolumeMap.h"
#include "SPHKernelFunc.cuh"

void DensityMatMulPosVecInvoker
(
	const Real* vParticlePosGPUPtr,
	const UInt* vNeighborCountGPUPtr,
	Real vStiffness,
	Real vDensity,
	UInt vConstraintSize,
	Real vMass,
	Real vDeltaT,
	CCuDenseVector& voResult,
	bool vIsMInv
);

void computeInertiaTermInvoker
(
	Real* voXn,
	const Real* vVn,
	const Real* vAn,
	UInt vParticleSize,
	Real vDeltaT
);

void initProjectiveSmoothKernelCubic(const CCubicKernel& SmoothKernelCubicCPU);

void solveDensityConstraintsInvoker
(
	const Real* vPos,
	const UInt* vNeighborData,
	const UInt* vNeighborCount,
	const UInt* vNeighborOffset,
	const vector<shared_ptr<CRigidBodyBoundaryVolumeMap>>& vBoundarys,
	UInt vConstraintSize,
	Real vParticleVolume,
	Real vStiffness,
	Real vDeltaTime,
	UInt vMaxIterationNum,
	Real vThreshold,
	CCuDenseVector& voVectorb
);

void computeAccelerationInvoker
(
	CCuDenseVector& voAcceleration,
	Real vGravity
);

Real updateTimeStepInvoker
(
	const CCuDenseVector& vParticleVel,
	const CCuDenseVector& vParticleAcc,
	UInt vParticleSize,
	Real vCFLFactor,
	Real vParticleRadius,
	Real vMaxTimeStepSize,
	Real vMinTimeStepSize
);

void solveLinerInvoker
(
	const thrust::device_vector<Real>& vVectorb,
	const thrust::device_vector<UInt>& vNeighborCount,
	Real vTimeStep2,
	Real vStiffness,
	Real vMass,

	thrust::device_vector<Real>& voX
);