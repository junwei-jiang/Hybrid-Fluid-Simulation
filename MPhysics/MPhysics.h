#pragma once
/*! \file MPhysics.h */
#ifdef MPHYSICS
#define MPHYSICS_API extern "C" __declspec(dllexport)
#else
#define MPHYSICS_API extern "C" __declspec(dllimport)
#endif

namespace MPhysicsAPI
{
	/*!
		初始化cuda context，包括：CUDA设备属性、默认Stream、Cublas和Cusparse等
	*/
	MPHYSICS_API void initMPhysics();

	/*!
		释放cuda context
	*/
	MPHYSICS_API void freeMPhysics();

	/*!
		前进vDeltaTime(s)
		\param [in] vDeltaTime 模拟的现实世界经过的时间
	*/
	MPHYSICS_API void update(float vDeltaTime);

	/*!
		前进vDeltaTime(s)
		\param [in] vDeltaTime 模拟的现实世界经过的时间
	*/
	MPHYSICS_API void loadScene(const char* vSceneFile);

#ifdef DOUBLE_REAL
	MPHYSICS_API void copyPositionToCPU(const char* vSimulatorName, double* voPosition);
	MPHYSICS_API void copyVelToCPU(const char* vSimulatorName, double* voVel);
	MPHYSICS_API void copyEulerParticlesPosToCPU(const char* vSimulatorName, double* voPosition);
	MPHYSICS_API void copyEulerParticlesVelToCPU(const char* vSimulatorName, double* voVel);
	MPHYSICS_API void copyFluidSDFToCPU(const char* vSimulatorName, double* voSDF);
	MPHYSICS_API void copyFluidSDFGradientToCPU(const char* vSimulatorName, double* voSDFGradientX, double* voSDFGradientY, double* voSDFGradientZ);
#else
	MPHYSICS_API void copyPositionToCPU(const char* vSimulatorName, float* voPosition);
	MPHYSICS_API void copyVelToCPU(const char* vSimulatorName, float* voVel);
	MPHYSICS_API void copyEulerParticlesPosToCPU(const char* vSimulatorName, float* voPosition);
	MPHYSICS_API void copyEulerParticlesVelToCPU(const char* vSimulatorName, float* voVel);
	MPHYSICS_API void copyFluidSDFToCPU(const char* vSimulatorName, float* voSDF);
	MPHYSICS_API void copyFluidSDFGradientToCPU(const char* vSimulatorName, float* voSDFGradientX, float* voSDFGradientY, float* voSDFGradientZ);
#endif // DOUBLE_REAL

	MPHYSICS_API unsigned int instanceRigidBoundary
	(
		const char* vSimulatorName, unsigned int vBoundaryIndex,
		float vPosX = 0.0f, float vPosY = 0.0f, float vPosZ = 0.0f,
		float R00 = 1.0f, float R01 = 0.0f, float R02 = 0.0f,
		float R10 = 0.0f, float R11 = 1.0f, float R12 = 0.0f,
		float R20 = 0.0f, float R21 = 0.0f, float R22 = 1.0f
	);

	MPHYSICS_API void transformRigidBoundary
	(
		const char* vSimulatorName, 
		unsigned int vBoundaryIndex,
		unsigned int vBoundaryInstanceIndex,
		float vPosX = 0.0f, float vPosY = 0.0f, float vPosZ = 0.0f,
		float R00 = 1.0f, float R01 = 0.0f, float R02 = 0.0f,
		float R10 = 0.0f, float R11 = 1.0f, float R12 = 0.0f,
		float R20 = 0.0f, float R21 = 0.0f, float R22 = 1.0f
	);
	MPHYSICS_API unsigned int getParticleSize(const char* vSimulatorName);
	MPHYSICS_API unsigned int getEulerParticleSize(const char* vSimulatorName);
	MPHYSICS_API unsigned int getNeighborDataSize(const char* vSimulatorName);
	MPHYSICS_API void copyParticleNeighborDataToCPU
	(
		const char* vSimulatorName, 
		unsigned int* voNeighborData, 
		unsigned int* voNeighborCount,
		unsigned int* voNeighborOffset
	);

	MPHYSICS_API void getGridInfo
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
	);

	MPHYSICS_API void getEulerGridInfo
	(
		const char* vSimulatorName,
		float& voGridOriginX, float& voGridOriginY, float& voGridOriginZ,
		float& voGridSpacingX, float& voGridSpacingY, float& voGridSpacingZ
	);

	MPHYSICS_API void copyCellDataToCPU
	(
		const char* vSimulatorName,
		unsigned int* voCellParticleCounts,
		unsigned int* voCellParticleOffsets
	);
};