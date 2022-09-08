#include "MPhysics.h"

#include "CudaContextManager.h"
#include "SceneManager.h"

void MPhysicsAPI::initMPhysics()
{
	initAgzUtilsDX11Device();
	CCudaContextManager::getInstance().initCudaContext();
}

void MPhysicsAPI::freeMPhysics()
{
	CSceneManager::getInstance().freeScene();
	CCudaContextManager::getInstance().freeCudaContext();
	freeAgzUtilsDX11Device();
}

void MPhysicsAPI::update(float vDeltaTime)
{
	for (auto HybridSimulators : CSceneManager::getInstance().getAllHybridSimulators())
	{
		HybridSimulators.second->update(vDeltaTime);
	}
}

void MPhysicsAPI::loadScene(const char* vSceneFile)
{
	CSceneManager::getInstance().loadScene(vSceneFile);
}

unsigned int MPhysicsAPI::instanceRigidBoundary
(
	const char* vSimulatorName, unsigned int vBoundaryIndex,
	float vPosX, float vPosY, float vPosZ,
	float R00, float R01, float R02,
	float R10, float R11, float R12,
	float R20, float R21, float R22
)
{
	shared_ptr<CHybridSimulator> Simulator = CSceneManager::getInstance().getHybridSimulators(vSimulatorName);
	SMatrix3x3 R;
	R.setCol(0, Vector3(R00, R10, R20));
	R.setCol(1, Vector3(R01, R11, R21));
	R.setCol(2, Vector3(R02, R12, R22));
	return Simulator->addRigidBoundaryInstance(vBoundaryIndex, R, Vector3(vPosX, vPosY, vPosZ));
}

void MPhysicsAPI::transformRigidBoundary
(
	const char* vSimulatorName, unsigned int vBoundaryIndex,
	unsigned int vBoundaryInstanceIndex,
	float vPosX, float vPosY, float vPosZ,
	float R00, float R01, float R02,
	float R10, float R11, float R12,
	float R20, float R21, float R22
)
{
	shared_ptr<CHybridSimulator> Simulator = CSceneManager::getInstance().getHybridSimulators(vSimulatorName);
	SMatrix3x3 R;
	R.setCol(0, Vector3(R00, R10, R20));
	R.setCol(1, Vector3(R01, R11, R21));
	R.setCol(2, Vector3(R02, R12, R22));
	Simulator->transformRigidBoundary(vBoundaryIndex, vBoundaryInstanceIndex, R, Vector3(vPosX, vPosY, vPosZ));
}

#ifdef DOUBLE_REAL
void MPhysicsAPI::copyPositionToCPU(const char * vSimulatorName, double * voPosition)
{
	shared_ptr<CParticleGroup> TargetParticleGroup = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getTargetParticleGroup();

	CHECK_CUDA(cudaMemcpy(
		voPosition,
		TargetParticleGroup->getConstParticlePosGPUPtr(),
		TargetParticleGroup->getSize() * 3 * sizeof(double),
		cudaMemcpyDeviceToHost
	));
}

void MPhysicsAPI::copyVelToCPU(const char * vSimulatorName, double * voVel)
{
	shared_ptr<CParticleGroup> TargetParticleGroup = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getTargetParticleGroup();

	CHECK_CUDA(cudaMemcpy(
		voVel,
		TargetParticleGroup->getConstParticleVelGPUPtr(),
		TargetParticleGroup->getSize() * 3 * sizeof(double),
		cudaMemcpyDeviceToHost
	));
}

void MPhysicsAPI::copyEulerParticlesPosToCPU(const char* vSimulatorName, double* voPosition)
{
	CEulerParticles EulerParticles = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getEulerParticles();

	CHECK_CUDA(cudaMemcpy(
		voPosition,
		EulerParticles.getConstParticlesPosGPUPtr(),
		EulerParticles.getNumOfParticles() * 3 * sizeof(double),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}

void MPhysicsAPI::copyEulerParticlesVelToCPU(const char* vSimulatorName, double* voVel)
{
	CEulerParticles EulerParticles = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getEulerParticles();

	CHECK_CUDA(cudaMemcpy(
		voVel,
		EulerParticles.getConstParticlesVelGPUPtr(),
		EulerParticles.getNumOfParticles() * 3 * sizeof(double),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}

void MPhysicsAPI::copyFluidSDFToCPU(const char* vSimulatorName, double* voSDF)
{
	CCellCenteredScalarField FluidSDF = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getFluidSDF();

	CHECK_CUDA(cudaMemcpy(
		voSDF,
		FluidSDF.getConstGridDataGPUPtr(),
		FluidSDF.getResolution().x * FluidSDF.getResolution().y * FluidSDF.getResolution().z * sizeof(double),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}
void MPhysicsAPI::copyFluidSDFGradientToCPU(const char* vSimulatorName, double* voSDFGradientX, double* voSDFGradientY, double* voSDFGradientZ)
{
	CCellCenteredScalarField FluidSDF = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getFluidSDF();
	CCellCenteredVectorField FluidSDFGradient(FluidSDF.getResolution(), FluidSDF.getOrigin(), FluidSDF.getSpacing());
	FluidSDF.gradient(FluidSDFGradient);

	CHECK_CUDA(cudaMemcpy(
		voSDFGradientX,
		FluidSDFGradient.getConstGridDataXGPUPtr(),
		FluidSDFGradient.getResolution().x * FluidSDFGradient.getResolution().y * FluidSDFGradient.getResolution().z * sizeof(double),
		cudaMemcpyDeviceToHost
	));
	CHECK_CUDA(cudaMemcpy(
		voSDFGradientY,
		FluidSDFGradient.getConstGridDataYGPUPtr(),
		FluidSDFGradient.getResolution().x * FluidSDFGradient.getResolution().y * FluidSDFGradient.getResolution().z * sizeof(double),
		cudaMemcpyDeviceToHost
	));
	CHECK_CUDA(cudaMemcpy(
		voSDFGradientZ,
		FluidSDFGradient.getConstGridDataZGPUPtr(),
		FluidSDFGradient.getResolution().x * FluidSDFGradient.getResolution().y * FluidSDFGradient.getResolution().z * sizeof(double),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}
#else
void MPhysicsAPI::copyPositionToCPU(const char * vSimulatorName, float * voPosition)
{
	shared_ptr<CParticleGroup> TargetParticleGroup = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getTargetParticleGroup();

	CHECK_CUDA(cudaMemcpy(
		voPosition,
		TargetParticleGroup->getConstParticlePosGPUPtr(),
		TargetParticleGroup->getSize() * 3 * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}

void MPhysicsAPI::copyVelToCPU(const char * vSimulatorName, float * voVel)
{
	shared_ptr<CParticleGroup> TargetParticleGroup = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getTargetParticleGroup();

	CHECK_CUDA(cudaMemcpy(
		voVel,
		TargetParticleGroup->getConstParticleVelGPUPtr(),
		TargetParticleGroup->getSize() * 3 * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}

void MPhysicsAPI::copyEulerParticlesPosToCPU(const char* vSimulatorName, float* voPosition)
{
	CEulerParticles EulerParticles = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getEulerParticles();

	CHECK_CUDA(cudaMemcpy(
		voPosition,
		EulerParticles.getConstParticlesPosGPUPtr(),
		EulerParticles.getNumOfParticles() * 3 * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}

void MPhysicsAPI::copyEulerParticlesVelToCPU(const char* vSimulatorName, float* voVel)
{
	CEulerParticles EulerParticles = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getEulerParticles();

	CHECK_CUDA(cudaMemcpy(
		voVel,
		EulerParticles.getConstParticlesVelGPUPtr(),
		EulerParticles.getNumOfParticles() * 3 * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}

void MPhysicsAPI::copyFluidSDFToCPU(const char* vSimulatorName, float* voSDF)
{
	CCellCenteredScalarField FluidSDF = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getFluidSDF();

	CHECK_CUDA(cudaMemcpy(
		voSDF,
		FluidSDF.getConstGridDataGPUPtr(),
		FluidSDF.getResolution().x * FluidSDF.getResolution().y * FluidSDF.getResolution().z * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}
void MPhysicsAPI::copyFluidSDFGradientToCPU(const char* vSimulatorName, float* voSDFGradientX, float* voSDFGradientY, float* voSDFGradientZ)
{
	CCellCenteredScalarField FluidSDF = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getFluidSDF();
	CCellCenteredVectorField FluidSDFGradient(FluidSDF.getResolution(), FluidSDF.getOrigin(), FluidSDF.getSpacing());
	FluidSDF.gradient(FluidSDFGradient);

	CHECK_CUDA(cudaMemcpy(
		voSDFGradientX,
		FluidSDFGradient.getConstGridDataXGPUPtr(),
		FluidSDFGradient.getResolution().x * FluidSDFGradient.getResolution().y * FluidSDFGradient.getResolution().z * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CHECK_CUDA(cudaMemcpy(
		voSDFGradientY,
		FluidSDFGradient.getConstGridDataYGPUPtr(),
		FluidSDFGradient.getResolution().x * FluidSDFGradient.getResolution().y * FluidSDFGradient.getResolution().z * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CHECK_CUDA(cudaMemcpy(
		voSDFGradientZ,
		FluidSDFGradient.getConstGridDataZGPUPtr(),
		FluidSDFGradient.getResolution().x * FluidSDFGradient.getResolution().y * FluidSDFGradient.getResolution().z * sizeof(float),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}
#endif // DOUBLE_REAL


unsigned int MPhysicsAPI::getParticleSize(const char * vSimulatorName)
{
	shared_ptr<CParticleGroup> TargetParticleGroup = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getTargetParticleGroup();
	return TargetParticleGroup->getSize();
}

unsigned int MPhysicsAPI::getEulerParticleSize(const char* vSimulatorName)
{
	CEulerParticles EulerParticles = CSceneManager::getInstance().getHybridSimulators(vSimulatorName)->getEulerParticles();
	return EulerParticles.getNumOfParticles();
}

unsigned int MPhysicsAPI::getNeighborDataSize(const char* vSimulatorName)
{
	shared_ptr<CHybridSimulator> Simulator = CSceneManager::getInstance().getHybridSimulators(vSimulatorName);
	return Simulator->getConstNeighborDataSize();
}

void MPhysicsAPI::copyParticleNeighborDataToCPU
(
	const char * vSimulatorName, 
	unsigned int * voNeighborData, 
	unsigned int * voNeighborCount,
	unsigned int * voNeighborOffset
)
{
	shared_ptr<CHybridSimulator> Simulator = CSceneManager::getInstance().getHybridSimulators(vSimulatorName);
	shared_ptr<CParticleGroup> TargetParticleGroup = Simulator->getTargetParticleGroup();
	CHECK_CUDA(cudaMemcpy(
		voNeighborData,
		Simulator->getConstNeighborDataGPUPtr(),
		Simulator->getConstNeighborDataSize() * sizeof(unsigned int),
		cudaMemcpyDeviceToHost
	)); 
	CUDA_SYN

	CHECK_CUDA(cudaMemcpy(
		voNeighborCount,
		Simulator->getConstNeighborCountGPUPtr(),
		TargetParticleGroup->getSize() * sizeof(unsigned int),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN

	CHECK_CUDA(cudaMemcpy(
		voNeighborOffset,
		Simulator->getConstNeighborOffsetGPUPtr(),
		TargetParticleGroup->getSize() * sizeof(unsigned int),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN

}

void MPhysicsAPI::getGridInfo
(
	const char* vSimulatorName,
	float& voGridMinX, float& voGridMinY, float& voGridMinZ,
	float& voGridDeltaX, float& voGridDeltaY, float& voGridDeltaZ,
	unsigned int& voGridDimX, unsigned int& voGridDimY, unsigned int& voGridDimZ,
	unsigned int& voMetaGridDimX, unsigned int& voMetaGridDimY, unsigned int& voMetaGridDimZ,
	unsigned int& voMetaGridDimGroupSize,
	unsigned int& voMetaGridDimBlockSize,
	unsigned int& voCellCount,
	float& voCellSize
)
{
	shared_ptr<CHybridSimulator> Simulator = CSceneManager::getInstance().getHybridSimulators(vSimulatorName);
	SGridInfo GridInfo = Simulator->getGridInfo();

	voGridMinX = GridInfo.AABB.Min.x;
	voGridMinY = GridInfo.AABB.Min.y;
	voGridMinZ = GridInfo.AABB.Min.z;

	voGridDeltaX = GridInfo.GridDelta.x;
	voGridDeltaY = GridInfo.GridDelta.y;
	voGridDeltaZ = GridInfo.GridDelta.z;

	voGridDimX = GridInfo.GridDimension.x;
	voGridDimY = GridInfo.GridDimension.y;
	voGridDimZ = GridInfo.GridDimension.z;

	voMetaGridDimX = GridInfo.MetaGridDimension.x;
	voMetaGridDimY = GridInfo.MetaGridDimension.y;
	voMetaGridDimZ = GridInfo.MetaGridDimension.z;

	voMetaGridDimGroupSize = GridInfo.MetaGridGroupSize;
	voMetaGridDimBlockSize = GridInfo.MetaGridBlockSize;

	voCellCount = Simulator->getCellDataSize();

	voCellSize = Simulator->getCellSize();
}

void MPhysicsAPI::getEulerGridInfo
(
	const char* vSimulatorName,
	float& voGridOriginX, float& voGridOriginY, float& voGridOriginZ,
	float& voGridSpacingX, float& voGridSpacingY, float& voGridSpacingZ
)
{
	shared_ptr<CHybridSimulator> Simulator = CSceneManager::getInstance().getHybridSimulators(vSimulatorName);

	voGridOriginX = Simulator->getGridOrigin().x;
	voGridOriginY = Simulator->getGridOrigin().y;
	voGridOriginZ = Simulator->getGridOrigin().z;

	voGridSpacingX = Simulator->getGridSpacing().x;
	voGridSpacingY = Simulator->getGridSpacing().y;
	voGridSpacingZ = Simulator->getGridSpacing().z;
}

void MPhysicsAPI::copyCellDataToCPU(const char * vSimulatorName, unsigned int * voCellParticleCounts, unsigned int * voCellParticleOffsets)
{
	shared_ptr<CHybridSimulator> Simulator = CSceneManager::getInstance().getHybridSimulators(vSimulatorName);

	CHECK_CUDA(cudaMemcpy(
		voCellParticleCounts,
		Simulator->getConstCellParticleCountsGPUPtr(),
		Simulator->getCellDataSize() * sizeof(unsigned int),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN

	CHECK_CUDA(cudaMemcpy(
		voCellParticleOffsets,
		Simulator->getConstCellParticleOffsetsGPUPtr(),
		Simulator->getCellDataSize() * sizeof(unsigned int),
		cudaMemcpyDeviceToHost
	));
	CUDA_SYN
}
