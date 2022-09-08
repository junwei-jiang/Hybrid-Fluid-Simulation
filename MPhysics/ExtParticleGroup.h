#pragma once
#include "Common.h"
#include "Particle.h"

class CExtParticleGroup : public CParticleGroup
{
public:
	CExtParticleGroup() = default;
	~CExtParticleGroup();

	void bindExtPosBuffer(void* vPosBufferHandle, UInt vBufferSizeInBytes);
	void bindPVSingle(void* vSemaphoreHandle);

	void wait(UInt vWaitValue);
	void signal(UInt vSignalValue);
	void copyToExtPosBuffer();

private:
	thrust::device_ptr<Real> m_ExtPosMemGPUPtr;
	cudaExternalMemory_t m_ExternalPosBuffer = nullptr;
	cudaExternalMemoryBufferDesc m_ExternalPosBufferDesc = {};
	cudaExternalSemaphore_t m_ExtSemaphore = nullptr;
};

