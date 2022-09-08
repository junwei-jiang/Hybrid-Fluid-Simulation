#include "EulerMathTool.cuh"

__device__ Real limitIndex(Real vIndex, Real vRes)
{
	if (vIndex < 0)
	{
		return 0;
	}
	if (vIndex > vRes - 1)
	{
		return vRes - 1;
	}
	return vIndex;
}

__device__ Int limitIndex(Int vIndex, Int vRes)
{
	if (vIndex < 0)
	{
		return 0;
	}
	if (vIndex > vRes - 1)
	{
		return vRes - 1;
	}
	return vIndex;
}

__device__ Vector3i transPos2Index
(
	Vector3 vParticlesPos,
	Vector3i vGridResolution,
	Vector3 vGridOrigin,
	Vector3 vGridSpacing
)
{
	Vector3 RelativePos = (vParticlesPos - vGridOrigin) / vGridSpacing;

	return Vector3i(limitIndex((Int)(RelativePos.x), vGridResolution.x), limitIndex((Int)(RelativePos.y), vGridResolution.y), limitIndex((Int)(RelativePos.z), vGridResolution.z));
}

__device__ bool isParticleInGrid
(
	Vector3 vParticlesPos,
	Vector3i vGridResolution,
	Vector3 vGridOrigin,
	Vector3 vGridSpacing
)
{
	Vector3 GridMax = vGridOrigin + Vector3(vGridResolution.x * vGridSpacing.x, vGridResolution.y * vGridSpacing.y, vGridResolution.z * vGridSpacing.z);

	if (vParticlesPos.x < vGridOrigin.x || vParticlesPos.y < vGridOrigin.y || vParticlesPos.z < vGridOrigin.z)
	{
		return false;
	}
	if (vParticlesPos.x > GridMax.x || vParticlesPos.y > GridMax.y || vParticlesPos.z > GridMax.z)
	{
		return false;
	}
	return true;
}

__device__ UInt transIndex2LinearWithOffset(Vector3i vIndex, Vector3i vRes, Vector3i vOffset)
{
	Int IndexX = vIndex.x + vOffset.x;
	Int IndexY = vIndex.y + vOffset.y;
	Int IndexZ = vIndex.z + vOffset.z;
	IndexX = limitIndex(IndexX, vRes.x);
	IndexY = limitIndex(IndexY, vRes.y);
	IndexZ = limitIndex(IndexZ, vRes.z);

	return (((IndexZ)* vRes.x * vRes.y) + ((IndexY)* vRes.x) + (IndexX));
}

__device__ Vector3i transLinearIndex2Coord(UInt vLinearIndex, Vector3i vRes)
{
	UInt IndexZ = vLinearIndex / (vRes.x * vRes.y);
	UInt IndexY = (vLinearIndex - IndexZ * (vRes.x * vRes.y)) / vRes.x;
	UInt IndexX = vLinearIndex % vRes.x;

	IndexX = limitIndex(IndexX, vRes.x);
	IndexY = limitIndex(IndexY, vRes.y);
	IndexZ = limitIndex(IndexZ, vRes.z);

	return Vector3i(IndexX, IndexY, IndexZ);
}

__device__ bool isRealEQ(Real vValueA, Real vValueB)
{
	return abs((vValueA - vValueB)) < REAL_MIN;
}

__device__ Int floorCUDA(Real vRealInput)
{
	if (vRealInput >= 0)
	{
		return ((Int)(vRealInput));
	}
	else
	{
		return (((Int)(vRealInput)) - 1);
	}
}

__device__ Real maxCUDA(Real vA, Real vB)
{
	if (vA > vB)
		return vA;
	else
		return vB;
}

__device__ Real minCUDA(Real vA, Real vB)
{
	if (vA < vB)
		return vA;
	else
		return vB;
}

__device__ Real clampCUDA(Real vInputValue, Real vA, Real vB)
{
	Real Max = maxCUDA(vA, vB);
	Real Min = minCUDA(vA, vB);

	return maxCUDA(Min, minCUDA(vInputValue, Max));
}

__device__ bool isInsideSDF(Real vInputSignedDistance)
{
	return vInputSignedDistance < 0;
}

__device__ Real cubicBridson(Real* vInputGPUPtr, Real vT)
{
	return (-1.0 / 3.0 * vT + 1.0 / 2.0 * vT * vT - 1.0 / 6.0 * vT * vT * vT) * vInputGPUPtr[0] +
		(1.0 - vT * vT + 1.0 / 2.0 * (vT * vT * vT - vT)) * vInputGPUPtr[1] +
		(vT + 1.0 / 2.0 * (vT * vT - vT * vT * vT)) * vInputGPUPtr[2] +
		(1.0 / 6.0 * (vT * vT * vT - vT)) * vInputGPUPtr[3];
}

__device__ Real biCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy)
{
	Real ValueY[4] = { cubicBridson(vInputGPUPtr, vTx), cubicBridson(vInputGPUPtr + 4, vTx), cubicBridson(vInputGPUPtr + 8, vTx), cubicBridson(vInputGPUPtr + 12, vTx) };
	return cubicBridson(ValueY, vTy);
}

__device__ Real triCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz)
{
	Real ValueZ[4] = { biCubicBridson(vInputGPUPtr, vTx, vTy), biCubicBridson(vInputGPUPtr + 16, vTx, vTy), biCubicBridson(vInputGPUPtr + 32, vTx, vTy), biCubicBridson(vInputGPUPtr + 48, vTx, vTy) };
	return cubicBridson(ValueZ, vTz);
}

__device__ Real clampCubicBridson(Real* vInputGPUPtr, Real vT)
{
	Real TempValue = (-1.0 / 3.0 * vT + 1.0 / 2.0 * vT * vT - 1.0 / 6.0 * vT * vT * vT) * vInputGPUPtr[0] +
		(1.0 - vT * vT + 1.0 / 2.0 * (vT * vT * vT - vT)) * vInputGPUPtr[1] +
		(vT + 1.0 / 2.0 * (vT * vT - vT * vT * vT)) * vInputGPUPtr[2] +
		(1.0 / 6.0 * (vT * vT * vT - vT)) * vInputGPUPtr[3];

	return clampCUDA(TempValue, vInputGPUPtr[1], vInputGPUPtr[2]);
}

__device__ Real biClampCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy)
{
	Real ValueY[4] = { clampCubicBridson(vInputGPUPtr, vTx), clampCubicBridson(vInputGPUPtr + 4, vTx), clampCubicBridson(vInputGPUPtr + 8, vTx), clampCubicBridson(vInputGPUPtr + 12, vTx) };
	return clampCubicBridson(ValueY, vTy);
}

__device__ Real triClampCubicBridson(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz)
{
	Real ValueZ[4] = { biClampCubicBridson(vInputGPUPtr, vTx, vTy), biClampCubicBridson(vInputGPUPtr + 16, vTx, vTy), biClampCubicBridson(vInputGPUPtr + 32, vTx, vTy), biClampCubicBridson(vInputGPUPtr + 48, vTx, vTy) };
	return clampCubicBridson(ValueZ, vTz);
}

__device__ Real catmullRom(Real* vInputGPUPtr, Real vT)
{
	Real Derivative1 = (vInputGPUPtr[2] - vInputGPUPtr[0]) / 2;
	Real Derivative2 = (vInputGPUPtr[3] - vInputGPUPtr[1]) / 2;
	Real Delta = vInputGPUPtr[2] - vInputGPUPtr[1];

	Real A3 = Derivative1 + Derivative2 - 2 * Delta;
	Real A2 = 3 * Delta - 2 * Derivative1 - Derivative2;
	Real A1 = Derivative1;
	Real A0 = vInputGPUPtr[1];

	return A3 * vT * vT * vT + A2 * vT * vT + A1 * vT + A0;
}

__device__ Real biCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy)
{
	Real ValueY[4] = { catmullRom(vInputGPUPtr, vTx), catmullRom(vInputGPUPtr + 4, vTx), catmullRom(vInputGPUPtr + 8, vTx), catmullRom(vInputGPUPtr + 12, vTx) };
	return catmullRom(ValueY, vTy);
}

__device__ Real triCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz)
{
	Real ValueZ[4] = { biCatmullRom(vInputGPUPtr, vTx, vTy), biCatmullRom(vInputGPUPtr + 16, vTx, vTy), biCatmullRom(vInputGPUPtr + 32, vTx, vTy), biCatmullRom(vInputGPUPtr + 48, vTx, vTy) };
	return catmullRom(ValueZ, vTz);
}

__device__ Real monotonicCatmullRom(Real* vInputGPUPtr, Real vT)
{
	Real Derivative1 = (vInputGPUPtr[2] - vInputGPUPtr[0]) / 2;
	Real Derivative2 = (vInputGPUPtr[3] - vInputGPUPtr[1]) / 2;
	Real Delta = vInputGPUPtr[2] - vInputGPUPtr[1];

	if (std::abs(Delta) < EPSILON)
	{
		Derivative1 = 0;
		Derivative2 = 0;
	}
	if (Delta * Derivative1 < 0)
	{
		Derivative1 = 0;
	}
	if (Delta * Derivative2 < 0)
	{
		Derivative2 = 0;
	}

	Real A3 = Derivative1 + Derivative2 - 2 * Delta;
	Real A2 = 3 * Delta - 2 * Derivative1 - Derivative2;
	Real A1 = Derivative1;
	Real A0 = vInputGPUPtr[1];

	return A3 * vT * vT * vT + A2 * vT * vT + A1 * vT + A0;
}

__device__ Real biMonotonicCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy)
{
	Real ValueY[4] = { monotonicCatmullRom(vInputGPUPtr, vTx), monotonicCatmullRom(vInputGPUPtr + 4, vTx), monotonicCatmullRom(vInputGPUPtr + 8, vTx), monotonicCatmullRom(vInputGPUPtr + 12, vTx) };
	return monotonicCatmullRom(ValueY, vTy);
}

__device__ Real triMonotonicCatmullRom(Real* vInputGPUPtr, Real vTx, Real vTy, Real vTz)
{
	Real ValueZ[4] = { biMonotonicCatmullRom(vInputGPUPtr, vTx, vTy), biMonotonicCatmullRom(vInputGPUPtr + 16, vTx, vTy), biMonotonicCatmullRom(vInputGPUPtr + 32, vTx, vTy), biMonotonicCatmullRom(vInputGPUPtr + 48, vTx, vTy) };
	return monotonicCatmullRom(ValueZ, vTz);
}

__device__ Vector3i transPos2GridIndex
(
	Vector3 vParticlesPos,
	Vector3i vGridResolution,
	Vector3 vGridOrigin,
	Vector3 vGridSpacing
)
{
	if (isParticleInGrid(vParticlesPos, vGridResolution, vGridOrigin, vGridSpacing))
	{
		Vector3 DownBackLeftIndex = (vParticlesPos - vGridOrigin) / vGridSpacing;
		return Vector3i((Int)(DownBackLeftIndex.x), (Int)(DownBackLeftIndex.y), (Int)(DownBackLeftIndex.z));
	}
	else
	{
		return Vector3i(-1, -1, -1);
	}
}

__global__ void kernelSetRandom(curandState *vCurandStates, long vClockForRand)
{
	Int CurIndexX = threadIdx.x;

	curand_init(vClockForRand, CurIndexX, 0, &vCurandStates[CurIndexX]);
}

__device__ Real linearKernelFunc(Real vOffset)
{
	if (abs(vOffset) < 1)
		return 1.0 - abs(vOffset);
	else
		return 0.0;
}

__device__ Real quadraticKernelFunc(Real vOffset)
{
	if (abs(vOffset) < 0.5)
		return 0.75 - abs(vOffset) * abs(vOffset);
	else if (abs(vOffset) >= 0.5 && abs(vOffset) < 1.5)
		return 0.5 * (1.5 - abs(vOffset)) * (1.5 - abs(vOffset));
	else
		return 0.0;
}

__device__ Real cubicKernelFunc(Real vOffset)
{
	if (abs(vOffset) < 1.0)
		return 0.5 * abs(vOffset) * abs(vOffset) * abs(vOffset) - abs(vOffset) * abs(vOffset) + 2.0 / 3.0;
	else if (abs(vOffset) >= 1.0 && abs(vOffset) < 2.0)
		return 1.0 / 6.0 * (2.0 - abs(vOffset)) * (2.0 - abs(vOffset)) * (2.0 - abs(vOffset));
	else
		return 0.0;
}

__device__ Real triLinearKernelFunc(Vector3 vOffset)
{
	return linearKernelFunc(vOffset.x) * linearKernelFunc(vOffset.y) * linearKernelFunc(vOffset.z);
}

__device__ Real triQuadraticKernelFunc(Vector3 vOffset)
{
	return quadraticKernelFunc(vOffset.x) * quadraticKernelFunc(vOffset.y) * quadraticKernelFunc(vOffset.z);
}

__device__ Real triCubicKernelFunc(Vector3 vOffset)
{
	return cubicKernelFunc(vOffset.x) * cubicKernelFunc(vOffset.y) * cubicKernelFunc(vOffset.z);
}

__device__ Real getCCSFieldTriLinearWeightAndValue
(
	Vector3i vResolution, 
	Vector3 vOrigin, 
	Vector3 vSpacing, 
	Vector3 vPosition, 
	Real* vSrcScalarFieldDataGPUPtr, 
	Real* voWeight, 
	Real* voValue
)
{
	Vector3 RelPos = (vPosition - vOrigin - vSpacing * 0.5);
	Vector3 RelPosIndex = RelPos / vSpacing;
	Vector3i DownBackLeftIndex = Vector3i(floorCUDA(RelPosIndex.x), floorCUDA(RelPosIndex.y), floorCUDA(RelPosIndex.z));
	Vector3 OffsetVector = (RelPos - Vector3((Real)(DownBackLeftIndex.x) * vSpacing.x, (Real)(DownBackLeftIndex.y) * vSpacing.y, (Real)(DownBackLeftIndex.z) * vSpacing.z)) / vSpacing;

	Real WeightX[2] = { linearKernelFunc(OffsetVector.x), linearKernelFunc(OffsetVector.x - 1.0) };
	Real WeightY[2] = { linearKernelFunc(OffsetVector.y), linearKernelFunc(OffsetVector.y - 1.0) };
	Real WeightZ[2] = { linearKernelFunc(OffsetVector.z), linearKernelFunc(OffsetVector.z - 1.0) };

	for (int z = 0; z < 2; z++)
	{
		for (int y = 0; y < 2; y++)
		{
			for (int x = 0; x < 2; x++)
			{
				voWeight[z * 2 * 2 + y * 2 + x] = WeightX[x] * WeightY[y] * WeightZ[z];
				voValue[z * 2 * 2 + y * 2 + x] = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vResolution, Vector3i(x, y, z))];
			}
		}
	}
}

__device__ bool isInsideAir(Real vFluidSDF, Real vSolidSDF)
{
	//这里其实不应该有等于号的，但是我粒子法流体域追踪的时候执行P2G之后其他地方被置0了
	return ((vFluidSDF >= 0) && (vSolidSDF >= 0));
}