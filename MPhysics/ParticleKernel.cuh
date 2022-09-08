#pragma once
#include "Common.h"

void reduceEmptyParticleInvoker
(
	thrust::device_vector<Real>& voParticlePos,
	thrust::device_vector<Real>& voPrevParticlePos,
	thrust::device_vector<Real>& voParticleVel,
	thrust::device_vector<Real>& voParticleLiveTime,
	UInt & voRealSize,
	UInt & voSize
);

void reduceParticleInvoker
(
	const thrust::device_vector<bool>& vFilterMap,
	thrust::device_vector<Real>& voParticlePos,
	thrust::device_vector<Real>& voPrevParticlePos,
	thrust::device_vector<Real>& voParticleVel,
	thrust::device_vector<Real>& voParticleLiveTime,
	UInt & voSize
);

void appendEmptyParticleInvoker
(
	thrust::device_vector<Real>& voParticlePos,
	thrust::device_vector<Real>& voPrevParticlePos,
	thrust::device_vector<Real>& voParticleVel,
	thrust::device_vector<Real>& voParticleLiveTime,
	UInt & voRealSize,
	Real vAppendSize,
	Real vSize
);