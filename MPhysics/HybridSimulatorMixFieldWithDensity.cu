#include "HybridSimulatorKernel.cuh"
#include "CellCenteredScalarField.h"
#include "FieldMathTool.cuh"
#include "CudaContextManager.h"

__global__ void mixFieldWithDensity
(
	const Real* vScalarFieldAGPUPtr,
	Real* vioScalarFieldBGPUPtr,
	const Real* vWeightFieldAGPUPtr,
	const Real* vWeightFieldBGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	if ((vWeightFieldAGPUPtr[CurLinearIndex] + vWeightFieldBGPUPtr[CurLinearIndex]) > EPSILON)
	{
		vioScalarFieldBGPUPtr[CurLinearIndex] =
			(vWeightFieldAGPUPtr[CurLinearIndex] * vScalarFieldAGPUPtr[CurLinearIndex] +
			 vWeightFieldBGPUPtr[CurLinearIndex] * vioScalarFieldBGPUPtr[CurLinearIndex]) /
			(vWeightFieldAGPUPtr[CurLinearIndex] + vWeightFieldBGPUPtr[CurLinearIndex]);
	}
	else
	{
		return;
	}
}

__global__ void mixFieldXWithDensity
(
	Vector3i vResolution,
	Vector3i vResolutionX,
	const Real* vVectorFieldAXGPUPtr,
	Real* vioScalarFieldBXGPUPtr,
	const Real* vWeightFieldAGPUPtr,
	const Real* vWeightFieldBGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Vector3i CurCoordIndexX = transLinearIndex2Coord(CurLinearIndex, vResolutionX);

	Int LeftLinearIndex = transIndex2LinearWithOffset(CurCoordIndexX, vResolution, Vector3i(-1, 0, 0));
	Int RightLinearIndex = transIndex2LinearWithOffset(CurCoordIndexX, vResolution);

	//这里少乘一个0.5结果一样，但是快一丢丢
	Real WeightA = (vWeightFieldAGPUPtr[LeftLinearIndex] + vWeightFieldAGPUPtr[RightLinearIndex]) * 0.5;
	Real WeightB = (vWeightFieldBGPUPtr[LeftLinearIndex] + vWeightFieldBGPUPtr[RightLinearIndex]) * 0.5;

	if (WeightA + WeightB > EPSILON)
		vioScalarFieldBXGPUPtr[CurLinearIndex] = (WeightA * vVectorFieldAXGPUPtr[CurLinearIndex] + WeightB * vioScalarFieldBXGPUPtr[CurLinearIndex]) / (WeightA + WeightB);
	else
		return;
}

__global__ void mixFieldYWithDensity
(
	Vector3i vResolution,
	Vector3i vResolutionY,
	const Real* vVectorFieldAYGPUPtr,
	Real* vioScalarFieldBYGPUPtr,
	const Real* vWeightFieldAGPUPtr,
	const Real* vWeightFieldBGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Vector3i CurCoordIndexY = transLinearIndex2Coord(CurLinearIndex, vResolutionY);

	Int DownLinearIndex = transIndex2LinearWithOffset(CurCoordIndexY, vResolution, Vector3i(0, -1, 0));
	Int UpLinearIndex = transIndex2LinearWithOffset(CurCoordIndexY, vResolution);

	//这里少乘一个0.5结果一样，但是快一丢丢
	Real WeightA = (vWeightFieldAGPUPtr[DownLinearIndex] + vWeightFieldAGPUPtr[UpLinearIndex]) * 0.5;
	Real WeightB = (vWeightFieldBGPUPtr[DownLinearIndex] + vWeightFieldBGPUPtr[UpLinearIndex]) * 0.5;

	if (WeightA + WeightB > EPSILON)
		vioScalarFieldBYGPUPtr[CurLinearIndex] = (WeightA * vVectorFieldAYGPUPtr[CurLinearIndex] + WeightB * vioScalarFieldBYGPUPtr[CurLinearIndex]) / (WeightA + WeightB);
	else
		return;
}

__global__ void mixFieldZWithDensity
(
	Vector3i vResolution,
	Vector3i vResolutionZ,
	const Real* vVectorFieldAZGPUPtr,
	Real* vioScalarFieldBZGPUPtr,
	const Real* vWeightFieldAGPUPtr,
	const Real* vWeightFieldBGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Vector3i CurCoordIndexZ = transLinearIndex2Coord(CurLinearIndex, vResolutionZ);

	Int BackLinearIndex = transIndex2LinearWithOffset(CurCoordIndexZ, vResolution, Vector3i(0, 0, -1));
	Int FrontLinearIndex = transIndex2LinearWithOffset(CurCoordIndexZ, vResolution);

	//这里少乘一个0.5结果一样，但是快一丢丢
	Real WeightA = (vWeightFieldAGPUPtr[BackLinearIndex] + vWeightFieldAGPUPtr[FrontLinearIndex]) * 0.5;
	Real WeightB = (vWeightFieldBGPUPtr[BackLinearIndex] + vWeightFieldBGPUPtr[FrontLinearIndex]) * 0.5;

	if (WeightA + WeightB > EPSILON)
		vioScalarFieldBZGPUPtr[CurLinearIndex] = (WeightA * vVectorFieldAZGPUPtr[CurLinearIndex] + WeightB * vioScalarFieldBZGPUPtr[CurLinearIndex]) / (WeightA + WeightB);
	else
		return;
}

void mixFieldWithDensityInvoker
(
	const CCellCenteredScalarField& vScalarFieldA,
	CCellCenteredScalarField& voScalarFieldB,
	const CCellCenteredScalarField& vWeightFieldA,
	const CCellCenteredScalarField& vWeightFieldB
)
{
	Vector3i Resolution = vScalarFieldA.getResolution();

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	_ASSERTE(Resolution == voScalarFieldB.getResolution());
	_ASSERTE(Resolution == vWeightFieldA.getResolution());
	_ASSERTE(Resolution == vWeightFieldB.getResolution());

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	mixFieldWithDensity LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
		(
			vScalarFieldA.getConstGridDataGPUPtr(),
			voScalarFieldB.getGridDataGPUPtr(),
			vWeightFieldA.getConstGridDataGPUPtr(),
			vWeightFieldB.getConstGridDataGPUPtr(),
			TotalThreadNum
		);
}

void mixFieldWithDensityInvoker
(
	const CCellCenteredVectorField& vVectorFieldA,
	CCellCenteredVectorField& voVectorFieldB,
	const CCellCenteredScalarField& vWeightFieldA,
	const CCellCenteredScalarField& vWeightFieldB
)
{
	Vector3i Resolution = vVectorFieldA.getResolution();

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	_ASSERTE(Resolution == voVectorFieldB.getResolution());
	_ASSERTE(Resolution == vWeightFieldA.getResolution());
	_ASSERTE(Resolution == vWeightFieldB.getResolution());

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	mixFieldWithDensity LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
		(
			vVectorFieldA.getConstGridDataXGPUPtr(),
			voVectorFieldB.getGridDataXGPUPtr(),
			vWeightFieldA.getConstGridDataGPUPtr(),
			vWeightFieldB.getConstGridDataGPUPtr(),
			TotalThreadNum
		);
	mixFieldWithDensity LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
		(
			vVectorFieldA.getConstGridDataYGPUPtr(),
			voVectorFieldB.getGridDataYGPUPtr(),
			vWeightFieldA.getConstGridDataGPUPtr(),
			vWeightFieldB.getConstGridDataGPUPtr(),
			TotalThreadNum
		);
	mixFieldWithDensity LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
		(
			vVectorFieldA.getConstGridDataZGPUPtr(),
			voVectorFieldB.getGridDataZGPUPtr(),
			vWeightFieldA.getConstGridDataGPUPtr(),
			vWeightFieldB.getConstGridDataGPUPtr(),
			TotalThreadNum
		);
}

void mixFieldWithDensityInvoker
(
	const CFaceCenteredVectorField& vVectorFieldA,
	CFaceCenteredVectorField& voVectorFieldB,
	const CCellCenteredScalarField& vWeightFieldA,
	const CCellCenteredScalarField& vWeightFieldB
)
{
	Vector3i Resolution = vVectorFieldA.getResolution();
	Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
	Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);

	Int TotalThreadNumX = (Int)(ResolutionX.x * ResolutionX.y * ResolutionX.z);
	Int TotalThreadNumY = (Int)(ResolutionY.x * ResolutionY.y * ResolutionY.z);
	Int TotalThreadNumZ = (Int)(ResolutionZ.x * ResolutionZ.y * ResolutionZ.z);

	_ASSERTE(Resolution == voVectorFieldB.getResolution());
	_ASSERTE(Resolution == vWeightFieldA.getResolution());
	_ASSERTE(Resolution == vWeightFieldB.getResolution());

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNumX, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlockX(BlockSize);
	dim3 NumBlockX(GridSize);
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNumY, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlockY(BlockSize);
	dim3 NumBlockY(GridSize);
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNumZ, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlockZ(BlockSize);
	dim3 NumBlockZ(GridSize);

	mixFieldXWithDensity LANCH_PARAM_1D_GB(NumBlockX, ThreadPerBlockX)
		(
			Resolution,
			ResolutionX,
			vVectorFieldA.getConstGridDataXGPUPtr(),
			voVectorFieldB.getGridDataXGPUPtr(),
			vWeightFieldA.getConstGridDataGPUPtr(),
			vWeightFieldB.getConstGridDataGPUPtr(),
			TotalThreadNumX
		);
	mixFieldYWithDensity LANCH_PARAM_1D_GB(NumBlockY, ThreadPerBlockY)
		(
			Resolution,
			ResolutionY,
			vVectorFieldA.getConstGridDataYGPUPtr(),
			voVectorFieldB.getGridDataYGPUPtr(),
			vWeightFieldA.getConstGridDataGPUPtr(),
			vWeightFieldB.getConstGridDataGPUPtr(),
			TotalThreadNumY
		);
	mixFieldZWithDensity LANCH_PARAM_1D_GB(NumBlockZ, ThreadPerBlockZ)
		(
			Resolution,
			ResolutionZ,
			vVectorFieldA.getConstGridDataZGPUPtr(),
			voVectorFieldB.getGridDataZGPUPtr(),
			vWeightFieldA.getConstGridDataGPUPtr(),
			vWeightFieldB.getConstGridDataGPUPtr(),
			TotalThreadNumZ
		);
}