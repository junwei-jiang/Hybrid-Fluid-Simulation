#include "FieldMathTool.cuh" 

__device__ Real sampleOneCellInCCSFieldTrilerp
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3 vSrcFieldSpacing,
	Vector3 vSrcFieldOrigin,
	Vector3 vSampledPos
)
{
	Vector3 SampledRelPos = vSampledPos - vSrcFieldOrigin - 0.5 * vSrcFieldSpacing;

	Vector3 TempDownBackLeftIndex = SampledRelPos / vSrcFieldSpacing;

	Vector3i DownBackLeftIndex = Vector3i(floorCUDA(TempDownBackLeftIndex.x), floorCUDA(TempDownBackLeftIndex.y), floorCUDA(TempDownBackLeftIndex.z));

	Real OffsetRatioX = (SampledRelPos.x - (DownBackLeftIndex.x * vSrcFieldSpacing.x)) / vSrcFieldSpacing.x;
	Real OffsetRatioY = (SampledRelPos.y - (DownBackLeftIndex.y * vSrcFieldSpacing.y)) / vSrcFieldSpacing.y;
	Real OffsetRatioZ = (SampledRelPos.z - (DownBackLeftIndex.z * vSrcFieldSpacing.z)) / vSrcFieldSpacing.z;

	Real UpBackLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 1, 0))];
	Real UpBackRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 1, 0))];
	Real UpFrontLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 1, 1))];
	Real UpFrontRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 1, 1))];
	Real DownBackLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 0, 0))];
	Real DownBackRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 0, 0))];
	Real DownFrontLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 0, 1))];
	Real DownFrontRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 0, 1))];

	return 
		DownBackLeft * (1 - OffsetRatioX) * (1 - OffsetRatioY) * (1 - OffsetRatioZ) +
		DownBackRight * OffsetRatioX * (1 - OffsetRatioY) * (1 - OffsetRatioZ) +
		UpBackLeft * (1 - OffsetRatioX) * OffsetRatioY * (1 - OffsetRatioZ) +
		UpBackRight * OffsetRatioX * OffsetRatioY * (1 - OffsetRatioZ) +
		DownFrontLeft * (1 - OffsetRatioX) * (1 - OffsetRatioY) * OffsetRatioZ +
		DownFrontRight * OffsetRatioX * (1 - OffsetRatioY) * OffsetRatioZ +
		UpFrontLeft * (1 - OffsetRatioX) * OffsetRatioY *OffsetRatioZ +
		UpFrontRight * OffsetRatioX * OffsetRatioY * OffsetRatioZ;
}

__device__ Real getCellCenteredScalarFieldValue
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3i vIndex
)
{
	return vSrcScalarFieldDataGPUPtr[vIndex.z * vSrcFieldResolution.x * vSrcFieldResolution.y + vIndex.y * vSrcFieldResolution.x + vIndex.x];
}

__global__ void sampleCellCenteredScalarFieldTrilerp
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Real* voDstDataGPUPtr,
	const Real* vSampledAbsPosDataXGPUPtr,
	const Real* vSampledAbsPosDataYGPUPtr,
	const Real* vSampledAbsPosDataZGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3 vSrcFieldSpacing,
	Vector3 vSrcFieldOrigin,
	Int vTotalThreadNum,
	Int vDstDataSpan = 1,
	Int vDstDataOffset = 0
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Real SampledRelPosX = vSampledAbsPosDataXGPUPtr[CurLinearIndex] - vSrcFieldOrigin.x - vSrcFieldSpacing.x / 2;
	Real SampledRelPosY = vSampledAbsPosDataYGPUPtr[CurLinearIndex] - vSrcFieldOrigin.y - vSrcFieldSpacing.y / 2;
	Real SampledRelPosZ = vSampledAbsPosDataZGPUPtr[CurLinearIndex] - vSrcFieldOrigin.z - vSrcFieldSpacing.z / 2;

	Real TempDownBackLeftIndexX = SampledRelPosX / vSrcFieldSpacing.x;
	Real TempDownBackLeftIndexY = SampledRelPosY / vSrcFieldSpacing.y;
	Real TempDownBackLeftIndexZ = SampledRelPosZ / vSrcFieldSpacing.z;

	Vector3i DownBackLeftIndex = Vector3i(floorCUDA(TempDownBackLeftIndexX), floorCUDA(TempDownBackLeftIndexY), floorCUDA(TempDownBackLeftIndexZ));

	Real OffsetRatioX = (SampledRelPosX - (DownBackLeftIndex.x * vSrcFieldSpacing.x)) / vSrcFieldSpacing.x;
	Real OffsetRatioY = (SampledRelPosY - (DownBackLeftIndex.y * vSrcFieldSpacing.y)) / vSrcFieldSpacing.y;
	Real OffsetRatioZ = (SampledRelPosZ - (DownBackLeftIndex.z * vSrcFieldSpacing.z)) / vSrcFieldSpacing.z;

	Real UpBackLeft     = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 1, 0))];
	Real UpBackRight    = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 1, 0))];
	Real UpFrontLeft    = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 1, 1))];
	Real UpFrontRight   = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 1, 1))];
	Real DownBackLeft   = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 0, 0))];
	Real DownBackRight  = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 0, 0))];
	Real DownFrontLeft  = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 0, 1))];
	Real DownFrontRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 0, 1))];

	voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] =
		DownBackLeft * (1 - OffsetRatioX) * (1 - OffsetRatioY) * (1 - OffsetRatioZ) +
		DownBackRight * OffsetRatioX * (1 - OffsetRatioY) * (1 - OffsetRatioZ) +
		UpBackLeft * (1 - OffsetRatioX) * OffsetRatioY * (1 - OffsetRatioZ) +
		UpBackRight * OffsetRatioX * OffsetRatioY * (1 - OffsetRatioZ) +
		DownFrontLeft * (1 - OffsetRatioX) * (1 - OffsetRatioY) * OffsetRatioZ +
		DownFrontRight * OffsetRatioX * (1 - OffsetRatioY) * OffsetRatioZ +
		UpFrontLeft * (1 - OffsetRatioX) * OffsetRatioY *OffsetRatioZ +
		UpFrontRight * OffsetRatioX * OffsetRatioY * OffsetRatioZ;
}

__global__ void sampleCellCenteredScalarFieldCubic
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Real* voDstDataGPUPtr,
	const Real* vSampledAbsPosDataXGPUPtr,
	const Real* vSampledAbsPosDataYGPUPtr,
	const Real* vSampledAbsPosDataZGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3 vSrcFieldSpacing,
	Vector3 vSrcFieldOrigin,
	Int vTotalThreadNum,
	ESamplingAlgorithm vCubicAlg,
	Int vDstDataSpan = 1,
	Int vDstDataOffset = 0
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Real SampledRelPosX = vSampledAbsPosDataXGPUPtr[CurLinearIndex] - vSrcFieldOrigin.x - vSrcFieldSpacing.x / 2;
	Real SampledRelPosY = vSampledAbsPosDataYGPUPtr[CurLinearIndex] - vSrcFieldOrigin.y - vSrcFieldSpacing.y / 2;
	Real SampledRelPosZ = vSampledAbsPosDataZGPUPtr[CurLinearIndex] - vSrcFieldOrigin.z - vSrcFieldSpacing.z / 2;

	Real TempDownBackLeftIndexX = SampledRelPosX / vSrcFieldSpacing.x;
	Real TempDownBackLeftIndexY = SampledRelPosY / vSrcFieldSpacing.y;
	Real TempDownBackLeftIndexZ = SampledRelPosZ / vSrcFieldSpacing.z;

	Vector3i DownBackLeftIndex = Vector3i(floorCUDA(TempDownBackLeftIndexX), floorCUDA(TempDownBackLeftIndexY), floorCUDA(TempDownBackLeftIndexZ));

	Real OffsetRatioX = (SampledRelPosX - (DownBackLeftIndex.x * vSrcFieldSpacing.x)) / vSrcFieldSpacing.x;
	Real OffsetRatioY = (SampledRelPosY - (DownBackLeftIndex.y * vSrcFieldSpacing.y)) / vSrcFieldSpacing.y;
	Real OffsetRatioZ = (SampledRelPosZ - (DownBackLeftIndex.z * vSrcFieldSpacing.z)) / vSrcFieldSpacing.z;

	Real Value[64];
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				Value[i * 4 * 4 + j * 4 + k] = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(k - 1, j - 1, i - 1))];
			}
		}
	}

	if (vCubicAlg == ESamplingAlgorithm::CATMULLROM)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triCatmullRom(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else if (vCubicAlg == ESamplingAlgorithm::MONOCATMULLROM)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triMonotonicCatmullRom(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else if (vCubicAlg == ESamplingAlgorithm::CUBICBRIDSON)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triCubicBridson(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else if (vCubicAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triClampCubicBridson(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else
	{

	}
}

__global__ void sampleCellCenteredScalarFieldTrilerp
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Real* voDstDataGPUPtr,
	const Real* vSampledAbsPosDataGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3 vSrcFieldSpacing,
	Vector3 vSrcFieldOrigin,
	Int vTotalThreadNum,
	Int vDstDataSpan = 1,
	Int vDstDataOffset = 0
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Real SampledRelPosX = vSampledAbsPosDataGPUPtr[3 * CurLinearIndex] - vSrcFieldOrigin.x - vSrcFieldSpacing.x / 2;
	Real SampledRelPosY = vSampledAbsPosDataGPUPtr[3 * CurLinearIndex + 1] - vSrcFieldOrigin.y - vSrcFieldSpacing.y / 2;
	Real SampledRelPosZ = vSampledAbsPosDataGPUPtr[3 * CurLinearIndex + 2] - vSrcFieldOrigin.z - vSrcFieldSpacing.z / 2;

	Real TempDownBackLeftIndexX = SampledRelPosX / vSrcFieldSpacing.x;
	Real TempDownBackLeftIndexY = SampledRelPosY / vSrcFieldSpacing.y;
	Real TempDownBackLeftIndexZ = SampledRelPosZ / vSrcFieldSpacing.z;

	Vector3i DownBackLeftIndex = Vector3i(floorCUDA(TempDownBackLeftIndexX), floorCUDA(TempDownBackLeftIndexY), floorCUDA(TempDownBackLeftIndexZ));

	Real OffsetRatioX = (SampledRelPosX - (DownBackLeftIndex.x * vSrcFieldSpacing.x)) / vSrcFieldSpacing.x;
	Real OffsetRatioY = (SampledRelPosY - (DownBackLeftIndex.y * vSrcFieldSpacing.y)) / vSrcFieldSpacing.y;
	Real OffsetRatioZ = (SampledRelPosZ - (DownBackLeftIndex.z * vSrcFieldSpacing.z)) / vSrcFieldSpacing.z;

	Real UpBackLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 1, 0))];
	Real UpBackRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 1, 0))];
	Real UpFrontLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 1, 1))];
	Real UpFrontRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 1, 1))];
	Real DownBackLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 0, 0))];
	Real DownBackRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 0, 0))];
	Real DownFrontLeft = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(0, 0, 1))];
	Real DownFrontRight = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(1, 0, 1))];

	voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] =
		DownBackLeft * (1 - OffsetRatioX) * (1 - OffsetRatioY) * (1 - OffsetRatioZ) +
		DownBackRight * OffsetRatioX * (1 - OffsetRatioY) * (1 - OffsetRatioZ) +
		UpBackLeft * (1 - OffsetRatioX) * OffsetRatioY * (1 - OffsetRatioZ) +
		UpBackRight * OffsetRatioX * OffsetRatioY * (1 - OffsetRatioZ) +
		DownFrontLeft * (1 - OffsetRatioX) * (1 - OffsetRatioY) * OffsetRatioZ +
		DownFrontRight * OffsetRatioX * (1 - OffsetRatioY) * OffsetRatioZ +
		UpFrontLeft * (1 - OffsetRatioX) * OffsetRatioY *OffsetRatioZ +
		UpFrontRight * OffsetRatioX * OffsetRatioY * OffsetRatioZ;
}

__global__ void sampleCellCenteredScalarFieldCubic
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Real* voDstDataGPUPtr,
	const Real* vSampledAbsPosDataGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3 vSrcFieldSpacing,
	Vector3 vSrcFieldOrigin,
	Int vTotalThreadNum,
	ESamplingAlgorithm vCubicAlg,
	Int vDstDataSpan = 1,
	Int vDstDataOffset = 0
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Real SampledRelPosX = vSampledAbsPosDataGPUPtr[3 * CurLinearIndex] - vSrcFieldOrigin.x - vSrcFieldSpacing.x / 2;
	Real SampledRelPosY = vSampledAbsPosDataGPUPtr[3 * CurLinearIndex + 1] - vSrcFieldOrigin.y - vSrcFieldSpacing.y / 2;
	Real SampledRelPosZ = vSampledAbsPosDataGPUPtr[3 * CurLinearIndex + 2] - vSrcFieldOrigin.z - vSrcFieldSpacing.z / 2;

	Real TempDownBackLeftIndexX = SampledRelPosX / vSrcFieldSpacing.x;
	Real TempDownBackLeftIndexY = SampledRelPosY / vSrcFieldSpacing.y;
	Real TempDownBackLeftIndexZ = SampledRelPosZ / vSrcFieldSpacing.z;

	Vector3i DownBackLeftIndex = Vector3i(floorCUDA(TempDownBackLeftIndexX), floorCUDA(TempDownBackLeftIndexY), floorCUDA(TempDownBackLeftIndexZ));

	Real OffsetRatioX = (SampledRelPosX - (DownBackLeftIndex.x * vSrcFieldSpacing.x)) / vSrcFieldSpacing.x;
	Real OffsetRatioY = (SampledRelPosY - (DownBackLeftIndex.y * vSrcFieldSpacing.y)) / vSrcFieldSpacing.y;
	Real OffsetRatioZ = (SampledRelPosZ - (DownBackLeftIndex.z * vSrcFieldSpacing.z)) / vSrcFieldSpacing.z;

	Real Value[64];
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				Value[i * 4 * 4 + j * 4 + k] = vSrcScalarFieldDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftIndex, vSrcFieldResolution, Vector3i(k - 1, j - 1, i - 1))];
			}
		}
	}

	if (vCubicAlg == ESamplingAlgorithm::CATMULLROM)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triCatmullRom(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else if (vCubicAlg == ESamplingAlgorithm::MONOCATMULLROM)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triMonotonicCatmullRom(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else if (vCubicAlg == ESamplingAlgorithm::CUBICBRIDSON)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triCubicBridson(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else if (vCubicAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		voDstDataGPUPtr[vDstDataSpan * CurLinearIndex + vDstDataOffset] = triClampCubicBridson(Value, OffsetRatioX, OffsetRatioY, OffsetRatioZ);
	}
	else
	{

	}
}

__global__ void CUDAGradient
(
	const Real* vScalarFieldDataGPUPtr,
	Real* voGradientFieldDataXGPUPtr,
	Real* voGradientFieldDataYGPUPtr,
	Real* voGradientFieldDataZGPUPtr,
	Vector3i vResolution,
	Vector3 vSpacing
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Real Left  = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0))];
	Real Right = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i( 1, 0, 0))];
	Real Down  = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0))];
	Real Up    = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0,  1, 0))];
	Real Back  = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1))];
	Real Front = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0,  1))];

	voGradientFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Right - Left) / vSpacing.x;
	voGradientFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Up - Down) / vSpacing.y;
	voGradientFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Front - Back) / vSpacing.z;
}

__global__ void CUDALaplacian
(
	const Real* vScalarFieldDataGPUPtr,
	Real* voLaplacianFieldDataGPUPtr,
	Vector3i vResolution,
	Vector3 vSpacing
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Real Center = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i( 0, 0, 0))];
	Real Left   = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0))];
	Real Right  = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i( 1, 0, 0))];
	Real Down   = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0))];
	Real Up     = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0,  1, 0))];
	Real Back   = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1))];
	Real Front  = vScalarFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0,  1))];

	voLaplacianFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 
		(Right - 2 * Center + Left) / (vSpacing.x * vSpacing.x) + 
		(Up    - 2 * Center + Down) / (vSpacing.y * vSpacing.y) +
		(Front - 2 * Center + Back) / (vSpacing.z * vSpacing.z);
}

__global__ void CUDADivergence
(
	const Real* vVectorFieldDataXGPUPtr,
	const Real* vVectorFieldDataYGPUPtr,
	const Real* vVectorFieldDataZGPUPtr,
	Real* voDivergenceFieldDataGPUPtr,
	Vector3i vResolution,
	Vector3 vSpacing
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Real Left  = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0))];
	Real Right = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i( 1, 0, 0))];
	Real Down  = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0))];
	Real Up    = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0,  1, 0))];
	Real Back  = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1))];
	Real Front = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0,  1))];

	voDivergenceFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Right - Left) / vSpacing.x + 0.5 * (Up - Down) / vSpacing.y + 0.5 * (Front - Back) / vSpacing.z;
}

__global__ void CUDADivergence
(
	const Real* vVectorFieldDataXGPUPtr,
	const Real* vVectorFieldDataYGPUPtr,
	const Real* vVectorFieldDataZGPUPtr,
	Real* voDivergenceFieldDataGPUPtr,
	Vector3i vResolution,
	Vector3i vResolutionX,
	Vector3i vResolutionY,
	Vector3i vResolutionZ,
	Vector3 vSpacing
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Real Left  = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(0, 0, 0))];
	Real Right = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(1, 0, 0))];
	Real Down  = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 0, 0))];
	Real Up    = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 1, 0))];
	Real Back  = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0, 0, 0))];
	Real Front = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0, 0, 1))];

	voDivergenceFieldDataGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = (Right - Left) / vSpacing.x + (Up - Down) / vSpacing.y + (Front - Back) / vSpacing.z;
}

__global__ void CUDACurl
(
	const Real* vVectorFieldDataXGPUPtr,
	const Real* vVectorFieldDataYGPUPtr,
	const Real* vVectorFieldDataZGPUPtr,
	Real* voCurlFieldDataXGPUPtr,
	Real* voCurlFieldDataYGPUPtr,
	Real* voCurlFieldDataZGPUPtr,
	Vector3i vResolution,
	Vector3 vSpacing
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Real Fx_ym = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0))];
	Real Fx_yp = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0,  1, 0))];
	Real Fx_zm = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1))];
	Real Fx_zp = vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0,  1))];

	Real Fy_xm = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0))];
	Real Fy_xp = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i( 1, 0, 0))];
	Real Fy_zm = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1))];
	Real Fy_zp = vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0,  1))];

	Real Fz_xm = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0))];
	Real Fz_xp = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i( 1, 0, 0))];
	Real Fz_ym = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0))];
	Real Fz_yp = vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0,  1, 0))];

	voCurlFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Fz_yp - Fz_ym) / vSpacing.y - 0.5 * (Fy_zp - Fy_zm) / vSpacing.z;
	voCurlFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Fx_zp - Fx_zm) / vSpacing.z - 0.5 * (Fz_xp - Fz_xm) / vSpacing.x;
	voCurlFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Fy_xp - Fy_xm) / vSpacing.x - 0.5 * (Fx_yp - Fx_ym) / vSpacing.y;
}

__global__ void CUDACurl
(
	const Real* vVectorFieldDataXGPUPtr,
	const Real* vVectorFieldDataYGPUPtr,
	const Real* vVectorFieldDataZGPUPtr,
	Real* voCurlFieldDataXGPUPtr,
	Real* voCurlFieldDataYGPUPtr,
	Real* voCurlFieldDataZGPUPtr,
	Vector3i vResolution,
	Vector3i vResolutionX,
	Vector3i vResolutionY,
	Vector3i vResolutionZ,
	Vector3 vSpacing
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Real Fx_ym = 0.5 * (vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(0, -1, 0))] + vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(1, -1, 0))]);
	Real Fx_yp = 0.5 * (vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(0,  1, 0))] + vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(1,  1, 0))]);
	Real Fx_zm = 0.5 * (vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(0, 0, -1))] + vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(1, 0, -1))]);
	Real Fx_zp = 0.5 * (vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(0, 0,  1))] + vVectorFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(1, 0,  1))]);

	Real Fy_xm = 0.5 * (vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(-1, 0, 0))] + vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(-1, 1, 0))]);
	Real Fy_xp = 0.5 * (vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i( 1, 0, 0))] + vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i( 1, 1, 0))]);
	Real Fy_zm = 0.5 * (vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 0, -1))] + vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 1, -1))]);
	Real Fy_zp = 0.5 * (vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 0,  1))] + vVectorFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 1,  1))]);

	Real Fz_xm = 0.5 * (vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(-1, 0, 0))] + vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(-1, 0, 1))]);
	Real Fz_xp = 0.5 * (vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i( 1, 0, 0))] + vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i( 1, 0, 1))]);
	Real Fz_ym = 0.5 * (vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0, -1, 0))] + vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0, -1, 1))]);
	Real Fz_yp = 0.5 * (vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0,  1, 0))] + vVectorFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0, 1,  1))]);

	voCurlFieldDataXGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Fz_yp - Fz_ym) / vSpacing.y - 0.5 * (Fy_zp - Fy_zm) / vSpacing.z;
	voCurlFieldDataYGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Fx_zp - Fx_zm) / vSpacing.z - 0.5 * (Fz_xp - Fz_xm) / vSpacing.x;
	voCurlFieldDataZGPUPtr[transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution)] = 0.5 * (Fy_xp - Fy_xm) / vSpacing.x - 0.5 * (Fx_yp - Fx_ym) / vSpacing.y;
}

void sampleCellCenteredScalarFieldInvoker
(
	const CCellCenteredScalarField& vSrcScalarField,
	CCellCenteredScalarField& voDstScalarField,
	const CCellCenteredVectorField& vSampledAbsPosVectorField,
	ESamplingAlgorithm vSamplingAlg
)
{
	Vector3i DstFieldResolution = voDstScalarField.getResolution();
	Vector3i SrcFieldResolution = vSrcScalarField.getResolution();
	Vector3  SrcFieldOrigin = vSrcScalarField.getOrigin();
	Vector3  SrcFieldSpacing = vSrcScalarField.getSpacing();
	Int TotalThreadNum = (Int)(DstFieldResolution.x * DstFieldResolution.y * DstFieldResolution.z);

	_ASSERTE(DstFieldResolution == vSampledAbsPosVectorField.getResolution());
	
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	if (vSamplingAlg == ESamplingAlgorithm::TRILINEAR)
	{
		sampleCellCenteredScalarFieldTrilerp<<<NumBlock, ThreadPerBlock>>>
			(
				vSrcScalarField.getConstGridDataGPUPtr(),
				voDstScalarField.getGridDataGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum
			);
	}
	else if (vSamplingAlg == ESamplingAlgorithm::CATMULLROM || vSamplingAlg == ESamplingAlgorithm::MONOCATMULLROM || vSamplingAlg == ESamplingAlgorithm::CUBICBRIDSON || vSamplingAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcScalarField.getConstGridDataGPUPtr(),
				voDstScalarField.getGridDataGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg
			);
	}
}

void sampleCellCenteredScalarFieldInvoker
(
	const CCellCenteredScalarField& vSrcScalarField,
	thrust::device_vector<Real>& voDstData,
	const thrust::device_vector<Real>& vSampledAbsPos,
	ESamplingAlgorithm vSamplingAlg
)
{
	_ASSERTE(voDstData.size() == vSampledAbsPos.size() / 3);
	Int TotalThreadNum = (Int)(vSampledAbsPos.size() / 3);

	Vector3i SrcFieldResolution = vSrcScalarField.getResolution();
	Vector3  SrcFieldOrigin = vSrcScalarField.getOrigin();
	Vector3  SrcFieldSpacing = vSrcScalarField.getSpacing();

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	if (vSamplingAlg == ESamplingAlgorithm::TRILINEAR)
	{
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcScalarField.getConstGridDataGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum
			);
	}
	else if (vSamplingAlg == ESamplingAlgorithm::CATMULLROM || vSamplingAlg == ESamplingAlgorithm::MONOCATMULLROM || vSamplingAlg == ESamplingAlgorithm::CUBICBRIDSON || vSamplingAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcScalarField.getConstGridDataGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg
			);
	}
}

void sampleCellCenteredVectorFieldInvoker
(
	const CCellCenteredVectorField& vSrcVectorField,
	CCellCenteredVectorField& voDstVectorField,
	const CCellCenteredVectorField& vSampledAbsPosVectorField,
	ESamplingAlgorithm vSamplingAlg
)
{
	Vector3i DstFieldResolution = voDstVectorField.getResolution();
	Vector3i SrcFieldResolution = vSrcVectorField.getResolution();
	Vector3  SrcFieldOrigin = vSrcVectorField.getOrigin();
	Vector3  SrcFieldSpacing = vSrcVectorField.getSpacing();
	Int TotalThreadNum = (Int)(DstFieldResolution.x * DstFieldResolution.y * DstFieldResolution.z);

	_ASSERTE(DstFieldResolution == vSampledAbsPosVectorField.getResolution());

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	if (vSamplingAlg == ESamplingAlgorithm::TRILINEAR)
	{
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				voDstVectorField.getGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum
			);	
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				voDstVectorField.getGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum
			);	
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				voDstVectorField.getGridDataZGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum
			);
	}
	else if (vSamplingAlg == ESamplingAlgorithm::CATMULLROM || vSamplingAlg == ESamplingAlgorithm::MONOCATMULLROM || vSamplingAlg == ESamplingAlgorithm::CUBICBRIDSON || vSamplingAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				voDstVectorField.getGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				voDstVectorField.getGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				voDstVectorField.getGridDataZGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg
			);
	}
}

void sampleCellCenteredVectorFieldInvoker
(
	const CCellCenteredVectorField& vSrcVectorField,
	thrust::device_vector<Real>& voDstData,
	const thrust::device_vector<Real>& vSampledAbsPos,
	ESamplingAlgorithm vSamplingAlg
)
{
	_ASSERTE(voDstData.size() == vSampledAbsPos.size());
	Int TotalThreadNum = (Int)(vSampledAbsPos.size() / 3);

	Vector3i SrcFieldResolution = vSrcVectorField.getResolution();
	Vector3  SrcFieldOrigin = vSrcVectorField.getOrigin();
	Vector3  SrcFieldSpacing = vSrcVectorField.getSpacing();

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	if (vSamplingAlg == ESamplingAlgorithm::TRILINEAR)
	{
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				3,
				0
			);
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				3,
				1
			);
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				3,
				2
			);

	}
	else if (vSamplingAlg == ESamplingAlgorithm::CATMULLROM || vSamplingAlg == ESamplingAlgorithm::MONOCATMULLROM || vSamplingAlg == ESamplingAlgorithm::CUBICBRIDSON || vSamplingAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg,
				3,
				0
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg,
				3,
				1
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolution,
				SrcFieldSpacing,
				SrcFieldOrigin,
				TotalThreadNum,
				vSamplingAlg,
				3,
				2
			);
	}
}

void sampleFaceCenteredVectorFieldInvoker
(
	const CFaceCenteredVectorField& vSrcVectorField,
	CCellCenteredVectorField& voDstVectorField,
	const CCellCenteredVectorField& vSampledAbsPosVectorField,
	ESamplingAlgorithm vSamplingAlg
)
{
	Vector3i DstFieldResolution = voDstVectorField.getResolution();
	Int TotalThreadNum = (Int)(DstFieldResolution.x * DstFieldResolution.y * DstFieldResolution.z);

	_ASSERTE(DstFieldResolution == vSampledAbsPosVectorField.getResolution());

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	Vector3i SrcFieldResolutionX = vSrcVectorField.getResolution() + Vector3i(1, 0, 0);
	Vector3  SrcFieldSpacingX = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginX = vSrcVectorField.getOrigin() - Vector3(vSrcVectorField.getSpacing().x / 2, 0, 0);
	Vector3i SrcFieldResolutionY = vSrcVectorField.getResolution() + Vector3i(0, 1, 0);
	Vector3  SrcFieldSpacingY = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginY = vSrcVectorField.getOrigin() - Vector3(0, vSrcVectorField.getSpacing().y / 2, 0);
	Vector3i SrcFieldResolutionZ = vSrcVectorField.getResolution() + Vector3i(0, 0, 1);
	Vector3  SrcFieldSpacingZ = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginZ = vSrcVectorField.getOrigin() - Vector3(0, 0, vSrcVectorField.getSpacing().z / 2);

	if (vSamplingAlg == ESamplingAlgorithm::TRILINEAR)
	{
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				voDstVectorField.getGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionX,
				SrcFieldSpacingX,
				SrcFieldOriginX,
				TotalThreadNum
			);	
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				voDstVectorField.getGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionY,
				SrcFieldSpacingY,
				SrcFieldOriginY,
				TotalThreadNum
			);
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				voDstVectorField.getGridDataZGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionZ,
				SrcFieldSpacingZ,
				SrcFieldOriginZ,
				TotalThreadNum
			);
	}
	else if (vSamplingAlg == ESamplingAlgorithm::CATMULLROM || vSamplingAlg == ESamplingAlgorithm::MONOCATMULLROM || vSamplingAlg == ESamplingAlgorithm::CUBICBRIDSON || vSamplingAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				voDstVectorField.getGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionX,
				SrcFieldSpacingX,
				SrcFieldOriginX,
				TotalThreadNum,
				vSamplingAlg
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				voDstVectorField.getGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionY,
				SrcFieldSpacingY,
				SrcFieldOriginY,
				TotalThreadNum,
				vSamplingAlg
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				voDstVectorField.getGridDataZGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionZ,
				SrcFieldSpacingZ,
				SrcFieldOriginZ,
				TotalThreadNum,
				vSamplingAlg
			);
	}
}

void sampleFaceCenteredVectorFieldInvoker
(
	const CFaceCenteredVectorField& vSrcVectorField,
	thrust::device_vector<Real>& voDstData,
	const thrust::device_vector<Real>& vSampledAbsPos,
	ESamplingAlgorithm vSamplingAlg
)
{
	_ASSERTE(voDstData.size() == vSampledAbsPos.size());
	Int TotalThreadNum = (Int)(vSampledAbsPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	Vector3i SrcFieldResolutionX = vSrcVectorField.getResolution() + Vector3i(1, 0, 0);
	Vector3  SrcFieldSpacingX = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginX = vSrcVectorField.getOrigin() - Vector3(vSrcVectorField.getSpacing().x / 2, 0, 0);
	Vector3i SrcFieldResolutionY = vSrcVectorField.getResolution() + Vector3i(0, 1, 0);
	Vector3  SrcFieldSpacingY = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginY = vSrcVectorField.getOrigin() - Vector3(0, vSrcVectorField.getSpacing().y / 2, 0);
	Vector3i SrcFieldResolutionZ = vSrcVectorField.getResolution() + Vector3i(0, 0, 1);
	Vector3  SrcFieldSpacingZ = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginZ = vSrcVectorField.getOrigin() - Vector3(0, 0, vSrcVectorField.getSpacing().z / 2);

	if (vSamplingAlg == ESamplingAlgorithm::TRILINEAR)
	{
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolutionX,
				SrcFieldSpacingX,
				SrcFieldOriginX,
				TotalThreadNum,
				3,
				0
			);
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolutionY,
				SrcFieldSpacingY,
				SrcFieldOriginY,
				TotalThreadNum,
				3,
				1
			);
		sampleCellCenteredScalarFieldTrilerp << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolutionZ,
				SrcFieldSpacingZ,
				SrcFieldOriginZ,
				TotalThreadNum,
				3,
				2
			);
	}
	else if (vSamplingAlg == ESamplingAlgorithm::CATMULLROM || vSamplingAlg == ESamplingAlgorithm::MONOCATMULLROM || vSamplingAlg == ESamplingAlgorithm::CUBICBRIDSON || vSamplingAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolutionX,
				SrcFieldSpacingX,
				SrcFieldOriginX,
				TotalThreadNum,
				vSamplingAlg,
				3,
				0
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolutionY,
				SrcFieldSpacingY,
				SrcFieldOriginY,
				TotalThreadNum,
				vSamplingAlg,
				3,
				1
			);
		sampleCellCenteredScalarFieldCubic << <NumBlock, ThreadPerBlock >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				getRawDevicePointerReal(voDstData),
				getReadOnlyRawDevicePointer(vSampledAbsPos),
				SrcFieldResolutionZ,
				SrcFieldSpacingZ,
				SrcFieldOriginZ,
				TotalThreadNum,
				vSamplingAlg,
				3,
				2
			);
		
	}
}

void sampleFaceCenteredVectorFieldInvoker
(
	const CFaceCenteredVectorField& vSrcVectorField,
	CFaceCenteredVectorField& voDstVectorField,
	const CCellCenteredVectorField& vSampledAbsPosXVectorField,
	const CCellCenteredVectorField& vSampledAbsPosYVectorField,
	const CCellCenteredVectorField& vSampledAbsPosZVectorField,
	ESamplingAlgorithm vSamplingAlg
)
{
	_ASSERTE((voDstVectorField.getResolution() + Vector3i(1, 0, 0)) == vSampledAbsPosXVectorField.getResolution());
	_ASSERTE((voDstVectorField.getResolution() + Vector3i(0, 1, 0)) == vSampledAbsPosYVectorField.getResolution());
	_ASSERTE((voDstVectorField.getResolution() + Vector3i(0, 0, 1)) == vSampledAbsPosZVectorField.getResolution());

	Vector3i DstFieldResolutionX = voDstVectorField.getResolution() + Vector3i(1, 0, 0);
	Vector3i SrcFieldResolutionX = vSrcVectorField.getResolution() + Vector3i(1, 0, 0);
	Vector3  SrcFieldSpacingX = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginX = vSrcVectorField.getOrigin() - Vector3(vSrcVectorField.getSpacing().x / 2, 0, 0);
	Vector3i DstFieldResolutionY = voDstVectorField.getResolution() + Vector3i(0, 1, 0);
	Vector3i SrcFieldResolutionY = vSrcVectorField.getResolution() + Vector3i(0, 1, 0);
	Vector3  SrcFieldSpacingY = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginY = vSrcVectorField.getOrigin() - Vector3(0, vSrcVectorField.getSpacing().y / 2, 0);
	Vector3i DstFieldResolutionZ = voDstVectorField.getResolution() + Vector3i(0, 0, 1);
	Vector3i SrcFieldResolutionZ = vSrcVectorField.getResolution() + Vector3i(0, 0, 1);
	Vector3  SrcFieldSpacingZ = vSrcVectorField.getSpacing();
	Vector3  SrcFieldOriginZ = vSrcVectorField.getOrigin() - Vector3(0, 0, vSrcVectorField.getSpacing().z / 2);

	Int TotalThreadNumX = (Int)(DstFieldResolutionX.x * DstFieldResolutionX.y * DstFieldResolutionX.z);
	Int TotalThreadNumY = (Int)(DstFieldResolutionY.x * DstFieldResolutionY.y * DstFieldResolutionY.z);
	Int TotalThreadNumZ = (Int)(DstFieldResolutionZ.x * DstFieldResolutionZ.y * DstFieldResolutionZ.z);
	UInt BlockSizeX, GridSizeX;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNumX, BlockSizeX, GridSizeX, 0.125);
	dim3 ThreadPerBlockX(BlockSizeX);
	dim3 NumBlockX(GridSizeX);
	UInt BlockSizeY, GridSizeY;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNumY, BlockSizeY, GridSizeY, 0.125);
	dim3 ThreadPerBlockY(BlockSizeY);
	dim3 NumBlockY(GridSizeY);
	UInt BlockSizeZ, GridSizeZ;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNumZ, BlockSizeZ, GridSizeZ, 0.125);
	dim3 ThreadPerBlockZ(BlockSizeZ);
	dim3 NumBlockZ(GridSizeZ);

	if (vSamplingAlg == ESamplingAlgorithm::TRILINEAR)
	{
		sampleCellCenteredScalarFieldTrilerp << <NumBlockX, ThreadPerBlockX >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				voDstVectorField.getGridDataXGPUPtr(),
				vSampledAbsPosXVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosXVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosXVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionX,
				SrcFieldSpacingX,
				SrcFieldOriginX,
				TotalThreadNumX
			);
		sampleCellCenteredScalarFieldTrilerp << <NumBlockY, ThreadPerBlockY >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				voDstVectorField.getGridDataYGPUPtr(),
				vSampledAbsPosYVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosYVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosYVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionY,
				SrcFieldSpacingY,
				SrcFieldOriginY,
				TotalThreadNumY
			);
		sampleCellCenteredScalarFieldTrilerp << <NumBlockZ, ThreadPerBlockZ >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				voDstVectorField.getGridDataZGPUPtr(),
				vSampledAbsPosZVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosZVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosZVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionZ,
				SrcFieldSpacingZ,
				SrcFieldOriginZ,
				TotalThreadNumZ
			);
	}
	else if (vSamplingAlg == ESamplingAlgorithm::CATMULLROM || vSamplingAlg == ESamplingAlgorithm::MONOCATMULLROM || vSamplingAlg == ESamplingAlgorithm::CUBICBRIDSON || vSamplingAlg == ESamplingAlgorithm::CLAMPCUBICBRIDSON)
	{
		sampleCellCenteredScalarFieldCubic << <NumBlockX, ThreadPerBlockX >> >
			(
				vSrcVectorField.getConstGridDataXGPUPtr(),
				voDstVectorField.getGridDataXGPUPtr(),
				vSampledAbsPosXVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosXVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosXVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionX,
				SrcFieldSpacingX,
				SrcFieldOriginX,
				TotalThreadNumX,
				vSamplingAlg
			);
		sampleCellCenteredScalarFieldCubic << <NumBlockY, ThreadPerBlockY >> >
			(
				vSrcVectorField.getConstGridDataYGPUPtr(),
				voDstVectorField.getGridDataYGPUPtr(),
				vSampledAbsPosYVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosYVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosYVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionY,
				SrcFieldSpacingY,
				SrcFieldOriginY,
				TotalThreadNumY,
				vSamplingAlg
			);
		sampleCellCenteredScalarFieldCubic << <NumBlockZ, ThreadPerBlockZ >> >
			(
				vSrcVectorField.getConstGridDataZGPUPtr(),
				voDstVectorField.getGridDataZGPUPtr(),
				vSampledAbsPosZVectorField.getConstGridDataXGPUPtr(),
				vSampledAbsPosZVectorField.getConstGridDataYGPUPtr(),
				vSampledAbsPosZVectorField.getConstGridDataZGPUPtr(),
				SrcFieldResolutionZ,
				SrcFieldSpacingZ,
				SrcFieldOriginZ,
				TotalThreadNumZ,
				vSamplingAlg
			);
	}
}

void gradientInvoker(const CCellCenteredScalarField& vScalarField, CCellCenteredVectorField& voGradientField)
{
	Vector3i Resolution = vScalarField.getResolution();
	Vector3  Spacing = vScalarField.getSpacing();

	_ASSERTE(Resolution == voGradientField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	CUDAGradient << <NumBlock, ThreadPerBlock >> >
		(
			vScalarField.getConstGridDataGPUPtr(),
			voGradientField.getGridDataXGPUPtr(),
			voGradientField.getGridDataYGPUPtr(),
			voGradientField.getGridDataZGPUPtr(),
			Resolution,
			Spacing
		);
}

void laplacianInvoker(const CCellCenteredScalarField& vScalarField, CCellCenteredScalarField& voLaplacianField)
{
	Vector3i Resolution = vScalarField.getResolution();
	Vector3  Spacing = vScalarField.getSpacing();

	_ASSERTE(Resolution == voLaplacianField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	CUDALaplacian << <NumBlock, ThreadPerBlock >> >
		(
			vScalarField.getConstGridDataGPUPtr(),
			voLaplacianField.getGridDataGPUPtr(),
			Resolution,
			Spacing
		);
}

void divergenceInvoker(const CCellCenteredVectorField& vVectorField, CCellCenteredScalarField& voDivergenceField)
{
	Vector3i Resolution = vVectorField.getResolution();
	Vector3  Spacing = vVectorField.getSpacing();

	_ASSERTE(Resolution == voDivergenceField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	CUDADivergence << <NumBlock, ThreadPerBlock >> >
		(
			vVectorField.getConstGridDataXGPUPtr(),
			vVectorField.getConstGridDataYGPUPtr(),
			vVectorField.getConstGridDataZGPUPtr(),
			voDivergenceField.getGridDataGPUPtr(),
			Resolution,
			Spacing
		);
}

void divergenceInvoker(const CFaceCenteredVectorField& vVectorField, CCellCenteredScalarField& voDivergenceField)
{
	Vector3i Resolution = vVectorField.getResolution();
	Vector3i ResolutionX = vVectorField.getResolution() + Vector3i(1, 0, 0);
	Vector3i ResolutionY = vVectorField.getResolution() + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = vVectorField.getResolution() + Vector3i(0, 0, 1);
	Vector3  Spacing = vVectorField.getSpacing();

	_ASSERTE(Resolution == voDivergenceField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	CUDADivergence << <NumBlock, ThreadPerBlock >> >
		(
			vVectorField.getConstGridDataXGPUPtr(),
			vVectorField.getConstGridDataYGPUPtr(),
			vVectorField.getConstGridDataZGPUPtr(),
			voDivergenceField.getGridDataGPUPtr(),
			Resolution,
			ResolutionX,
			ResolutionY,
			ResolutionZ,
			Spacing
		);
}

void curlInvoker(const CCellCenteredVectorField& vVectorField, CCellCenteredVectorField& voCurlField)
{
	Vector3i Resolution = vVectorField.getResolution();
	Vector3  Spacing = vVectorField.getSpacing();

	_ASSERTE(Resolution == voCurlField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	CUDACurl << <NumBlock, ThreadPerBlock >> >
		(
			vVectorField.getConstGridDataXGPUPtr(),
			vVectorField.getConstGridDataYGPUPtr(),
			vVectorField.getConstGridDataZGPUPtr(),
			voCurlField.getGridDataXGPUPtr(),
			voCurlField.getGridDataYGPUPtr(),
			voCurlField.getGridDataZGPUPtr(),
			Resolution,
			Spacing
		);
}

void curlInvoker(const CFaceCenteredVectorField& vVectorField, CCellCenteredVectorField& voCurlField)
{
	Vector3i Resolution = vVectorField.getResolution();
	Vector3i ResolutionX = vVectorField.getResolution() + Vector3i(1, 0, 0);
	Vector3i ResolutionY = vVectorField.getResolution() + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = vVectorField.getResolution() + Vector3i(0, 0, 1);
	Vector3  Spacing = vVectorField.getSpacing();

	_ASSERT(Resolution == voCurlField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	CUDACurl << <NumBlock, ThreadPerBlock >> >
		(
			vVectorField.getConstGridDataXGPUPtr(),
			vVectorField.getConstGridDataYGPUPtr(),
			vVectorField.getConstGridDataZGPUPtr(),
			voCurlField.getGridDataXGPUPtr(),
			voCurlField.getGridDataYGPUPtr(),
			voCurlField.getGridDataZGPUPtr(),
			Resolution,
			ResolutionX,
			ResolutionY,
			ResolutionZ,
			Spacing
		);
}