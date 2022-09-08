#include "ExtParticleGroup.h"
#include "ThrustWapper.cuh"
#include <vector>
#include <cstdlib>
#include <ctime>
#include "CudaContextManager.h"

CExtParticleGroup::~CExtParticleGroup()
{
	cudaDestroyExternalSemaphore(m_ExtSemaphore);
	cudaDestroyExternalMemory(m_ExternalPosBuffer);
}

void CExtParticleGroup::wait(UInt vWaitValue)
{
	_ASSERT(m_ExtSemaphore != nullptr);

	cudaExternalSemaphoreWaitParams  WaitParams = {};

	memset(&WaitParams, 0, sizeof(WaitParams));
	WaitParams.params.fence.value = vWaitValue;
	WaitParams.flags = 0;

	CHECK_CUDA(cudaWaitExternalSemaphoresAsync(&m_ExtSemaphore, &WaitParams, 1));
}

void CExtParticleGroup::signal(UInt vSignalValue)
{
	_ASSERT(m_ExtSemaphore != nullptr);

	cudaExternalSemaphoreSignalParams SignalParams = {};

	memset(&SignalParams, 0, sizeof(SignalParams));
	SignalParams.params.fence.value = vSignalValue;
	SignalParams.flags = 0;

	CHECK_CUDA(cudaSignalExternalSemaphoresAsync(&m_ExtSemaphore, &SignalParams, 1));
}

void CExtParticleGroup::bindExtPosBuffer(void* vPosBufferHandle, UInt vBufferSizeInBytes)
{
	m_Size = vBufferSizeInBytes / (sizeof(Real) * 3);

	//Bind Position Buffer
	cudaExternalMemoryHandleDesc PosBufferExtMemHandleDesc = {};
	memset(&PosBufferExtMemHandleDesc, 0, sizeof(PosBufferExtMemHandleDesc));
	PosBufferExtMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
	PosBufferExtMemHandleDesc.handle.win32.handle = vPosBufferHandle;
	PosBufferExtMemHandleDesc.size = vBufferSizeInBytes;
	PosBufferExtMemHandleDesc.flags = cudaExternalMemoryDedicated;
	CHECK_CUDA(cudaImportExternalMemory(&m_ExternalPosBuffer, &PosBufferExtMemHandleDesc));
	
	memset(&m_ExternalPosBufferDesc, 0, sizeof(m_ExternalPosBufferDesc));
	m_ExternalPosBufferDesc.offset = 0;
	m_ExternalPosBufferDesc.size = vBufferSizeInBytes;
	m_ExternalPosBufferDesc.flags = 0;
	Real* RawParticlePosGPUPtr = nullptr;
	CHECK_CUDA(cudaExternalMemoryGetMappedBuffer((void**)&RawParticlePosGPUPtr, m_ExternalPosBuffer, &m_ExternalPosBufferDesc));
	m_ExtPosMemGPUPtr = thrust::device_ptr<Real>(RawParticlePosGPUPtr);
}

void CExtParticleGroup::bindPVSingle(void* vSemaphoreHandle)
{
	cudaExternalSemaphoreHandleDesc SemaphoreHandleDesc = {};
	memset(&SemaphoreHandleDesc, 0, sizeof(SemaphoreHandleDesc));

	SemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
	SemaphoreHandleDesc.handle.win32.handle = vSemaphoreHandle;

	CHECK_CUDA(cudaImportExternalSemaphore(&m_ExtSemaphore, &SemaphoreHandleDesc));
}

void CExtParticleGroup::copyToExtPosBuffer()
{
	_ASSERT(m_ExternalPosBuffer != nullptr);
	_ASSERT(m_ExtPosMemGPUPtr != nullptr);
	_ASSERT(m_ExtSemaphore != nullptr);

	CHECK_CUDA(cudaMemcpy(
		getRawDevicePointerReal(m_ExtPosMemGPUPtr),
		getRawDevicePointerReal(m_ParticlePos.getVectorValue()),
		m_Size * sizeof(Vector3),
		cudaMemcpyDeviceToDevice
	));
}