#pragma once
#include "CuDenseVector.h"
#include "EulerParticles.h"
#include "CellCenteredScalarField.h"
#include "CellCenteredVectorField.h"
#include "FaceCenteredVectorField.h"
#include "CudaContextManager.h"
#include <curand_kernel.h>

__device__ Real limitIndex(Real vIndex, Real vRes);

__device__ Int limitIndex(Int vIndex, Int vRes);

__device__ Vector3i transPos2Index
(
	Vector3 vParticlesPos,
	Vector3i vGridResolution,
	Vector3 vGridOrigin,
	Vector3 vGridSpacing
);

__device__ bool isParticleInGrid
(
	Vector3 vParticlesPos,
	Vector3i vGridResolution,
	Vector3 vGridOrigin,
	Vector3 vGridSpacing
);//未测试

__device__ UInt transIndex2LinearWithOffset(Vector3i vIndex, Vector3i vRes, Vector3i vOffset = Vector3i(0, 0, 0));

__device__ Vector3i transLinearIndex2Coord(UInt vLinearIndex, Vector3i vRes);

__device__ bool isRealEQ(Real vValueA, Real vValueB);

__device__ Int floorCUDA(Real vRealInput);

__device__ Real maxCUDA(Real vA, Real vB);

__device__ Real minCUDA(Real vA, Real vB);

__device__ Real clampCUDA(Real vInputValue, Real vA, Real vB);

__device__ bool isInsideSDF(Real vInputSignedDistance);

__device__ Real cubicBridson(Real* vInputGPUPtr, Real vT);

__device__ Real biCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy);

__device__ Real triCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz);

__device__ Real clampCubicBridson(Real* vInputGPUPtr, Real vT);

__device__ Real biClampCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy);

__device__ Real triClampCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz);

__device__ Real catmullRom(Real* vInputGPUPtr, Real vT);

__device__ Real biCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy);

__device__ Real triCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz);

__device__ Real monotonicCatmullRom(Real* vInputGPUPtr, Real vT);

__device__ Real biMonotonicCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy);

__device__ Real triMonotonicCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz);

__global__ void kernelSetRandom(curandState *vCurandStates, long vClockForRand);//未测试

__device__ Real linearKernelFunc(Real vOffset);

__device__ Real quadraticKernelFunc(Real vOffset);

__device__ Real cubicKernelFunc(Real vOffset);

__device__ Real triLinearKernelFunc(Vector3 vOffset);

__device__ Real triQuadraticKernelFunc(Vector3 vOffset);

__device__ Real triCubicKernelFunc(Vector3 vOffset);

__device__ Real getCCSFieldTriLinearWeightAndValue
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Vector3 vPosition,
	Real* vSrcScalarFieldDataGPUPtr,
	Real* voWeight,
	Real* voValue
);

__device__ bool isInsideAir(Real vFluidSDF, Real vSolidSDF);