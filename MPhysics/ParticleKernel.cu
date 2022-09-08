#include "ParticleKernel.cuh"

void reduceEmptyParticleInvoker
(
	thrust::device_vector<Real>& voParticlePos, 
	thrust::device_vector<Real>& voPrevParticlePos, 
	thrust::device_vector<Real>& voParticleVel,
	thrust::device_vector<Real>& voParticleLiveTime,
	UInt & voRealSize, 
	UInt & voSize
)
{
	UInt NewPosSize = 
		(thrust::remove(voParticlePos.begin() + voSize * 3, voParticlePos.begin() + voRealSize * 3, REAL_MAX) - voParticlePos.begin()) / 3;
	UInt NewPrevPosSize =
		(thrust::remove(voPrevParticlePos.begin() + voSize * 3, voPrevParticlePos.begin() + voRealSize * 3, REAL_MAX) - voPrevParticlePos.begin()) / 3;
	UInt NewVelSize =
		(thrust::remove(voParticleVel.begin() + voSize * 3, voParticleVel.begin() + voRealSize * 3, REAL_MAX) - voParticleVel.begin()) / 3;
	UInt NewLiveTimeSize =
		(thrust::remove(voParticleLiveTime.begin() + voSize * 3, voParticleLiveTime.begin() + voRealSize * 3, REAL_MAX) - voParticleLiveTime.begin()) / 3;
	_ASSERT(NewPosSize == NewPrevPosSize && NewPrevPosSize == NewVelSize && NewVelSize == NewLiveTimeSize);

	voParticlePos.resize(NewPosSize * 3);
	voPrevParticlePos.resize(NewPosSize * 3);
	voParticleVel.resize(NewPosSize * 3);
	voParticleLiveTime.resize(NewPosSize * 3);

	voSize = NewPosSize;
}

void reduceParticleInvoker
(
	const thrust::device_vector<bool>& vFilterMap, 
	thrust::device_vector<Real>& voParticlePos, 
	thrust::device_vector<Real>& voPrevParticlePos, 
	thrust::device_vector<Real>& voParticleVel, 
	thrust::device_vector<Real>& voParticleLiveTime, 
	UInt & voSize
)
{
	UInt NewPosSize =
		(thrust::remove_if(voParticlePos.begin(), voParticlePos.begin() + voSize * 3, vFilterMap.begin(), thrust::identity<bool>()) - voParticlePos.begin()) / 3;
	UInt NewPrevPosSize =
		(thrust::remove_if(voPrevParticlePos.begin(), voPrevParticlePos.begin() + voSize * 3, vFilterMap.begin(), thrust::identity<bool>()) - voPrevParticlePos.begin()) / 3;
	UInt NewVelSize =
		(thrust::remove_if(voParticleVel.begin(), voParticleVel.begin() + voSize * 3, vFilterMap.begin(), thrust::identity<bool>()) - voParticleVel.begin()) / 3;
	UInt NewLiveTimeSize =
		(thrust::remove_if(voParticleLiveTime.begin(), voParticleLiveTime.begin() + voSize * 3, vFilterMap.begin(), thrust::identity<bool>()) - voParticleLiveTime.begin()) / 3;
	_ASSERT(NewPosSize == NewPrevPosSize && NewPrevPosSize == NewVelSize && NewVelSize == NewLiveTimeSize);

	voParticlePos.resize(NewPosSize * 3);
	voPrevParticlePos.resize(NewPosSize * 3);
	voParticleVel.resize(NewPosSize * 3);
	voParticleLiveTime.resize(NewPosSize * 3);

	voSize = NewPosSize;
}

void appendEmptyParticleInvoker
(
	thrust::device_vector<Real>& voParticlePos, 
	thrust::device_vector<Real>& voPrevParticlePos, 
	thrust::device_vector<Real>& voParticleVel, 
	thrust::device_vector<Real>& voParticleLiveTime, 
	UInt & voRealSize,
	Real vAppendSize,
	Real vSize
)
{
	if (voRealSize - vSize >= vAppendSize)
	{
		return;
	}
	else
	{
		UInt AddSize = vAppendSize - (voRealSize - vSize);
		voParticlePos.resize((voRealSize + AddSize) * 3, REAL_MAX);
		voPrevParticlePos.resize((voRealSize + AddSize) * 3, REAL_MAX);
		voParticleVel.resize((voRealSize + AddSize) * 3, REAL_MAX);
		voParticleLiveTime.resize((voRealSize + AddSize) * 3, REAL_MAX);
		thrust::fill(voParticlePos.begin() + vSize * 3, voParticlePos.begin() + voRealSize * 3, REAL_MAX);
		thrust::fill(voPrevParticlePos.begin() + vSize * 3, voPrevParticlePos.begin() + voRealSize * 3, REAL_MAX);
		thrust::fill(voParticleVel.begin() + vSize * 3, voParticleVel.begin() + voRealSize * 3, REAL_MAX);
		thrust::fill(voParticleLiveTime.begin() + vSize * 3, voParticleLiveTime.begin() + voRealSize * 3, REAL_MAX);
		voRealSize = vAppendSize + vSize;
	}
}
