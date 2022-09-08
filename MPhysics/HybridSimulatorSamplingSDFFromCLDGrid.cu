#include "HybridSimulatorKernel.cuh"
#include "CLDGridinterpolate.cuh"
#include "EulerMathTool.cuh"
#include "CubicLagrangeDiscreteGrid.h"
#include "CudaContextManager.h"

__global__ void samplingSDFFromCLDGrid
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Real* voSDFGridDataGPUPtr,
	const Real* vNodeGPUPtr,
	const UInt* vCellGPUPtr,
	SCLDGridInfo vCLDGridInfo,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Vector3i CurCoordIndex = transLinearIndex2Coord(CurLinearIndex, vResolution);
	Vector3 CurCellPos = vOrigin + 0.5 * vSpacing + Vector3(vSpacing.x * CurCoordIndex.x, vSpacing.y * CurCoordIndex.y, vSpacing.z * CurCoordIndex.z);

	interpolateSinglePos
	(
		CurCellPos,
		vNodeGPUPtr,
		vCellGPUPtr,
		vCLDGridInfo,
		voSDFGridDataGPUPtr[CurLinearIndex]
	);
}

void samplingSDFFromCLDGridInvoker
(
	std::shared_ptr<CCubicLagrangeDiscreteGrid> vCLDGrid,
	CCellCenteredScalarField& voSDFField
)
{
	Vector3i Resolution = voSDFField.getResolution();
	Vector3 Origin = voSDFField.getOrigin();
	Vector3 Spacing = voSDFField.getSpacing();

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	samplingSDFFromCLDGrid LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
		(
			Resolution,
			Origin,
			Spacing,
			voSDFField.getGridDataGPUPtr(),
			getReadOnlyRawDevicePointer(vCLDGrid->getField(0)),
			getReadOnlyRawDevicePointer(vCLDGrid->getCell()),
			vCLDGrid->getCLDGridInfo(),
			TotalThreadNum
		);
}