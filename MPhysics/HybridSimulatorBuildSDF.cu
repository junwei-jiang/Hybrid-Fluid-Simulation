#include "HybridSimulatorKernel.cuh"
#include "CudaContextManager.h"

__global__ void buildFluidInsideSDF
(
	Vector3i vResolution,
	const Real* vFluidDensityFieldDataGPUPtr,
	const Real* vSolidDomainFieldDataGPUPtr,
	Real* vioFluidInsideSDFFieldDataGPUPtr,
	UInt vCurrentDis
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CenterLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);
	Int LeftLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0));
	Int RightLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(1, 0, 0));
	Int DownLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0));
	Int UpLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 1, 0));
	Int BackLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1));
	Int FrontLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, 1));

	if (vCurrentDis == 0)
	{
		if (isInsideSDF(vSolidDomainFieldDataGPUPtr[CenterLinearIndex]))
		{
			vioFluidInsideSDFFieldDataGPUPtr[CenterLinearIndex] = 0.0;
		}
		else
		{
			if (isRealEQ(vFluidDensityFieldDataGPUPtr[CenterLinearIndex], FluidSurfaceDensity))
			{
				vioFluidInsideSDFFieldDataGPUPtr[CenterLinearIndex] = (FluidSurfaceDensity - 0.5);
			}
			else
			{
				if (isRealEQ(vFluidDensityFieldDataGPUPtr[CenterLinearIndex], 0.0))
				{
					vioFluidInsideSDFFieldDataGPUPtr[CenterLinearIndex] = 0.0;
				}
				else
				{
					vioFluidInsideSDFFieldDataGPUPtr[CenterLinearIndex] = UNKNOWN;
				}
			}
		}
	}
	else
	{
		if (vioFluidInsideSDFFieldDataGPUPtr[CenterLinearIndex] == UNKNOWN)
		{
			if (isRealEQ(vioFluidInsideSDFFieldDataGPUPtr[LeftLinearIndex], (vCurrentDis + (FluidSurfaceDensity - 0.5) - 1)) ||
				isRealEQ(vioFluidInsideSDFFieldDataGPUPtr[RightLinearIndex], (vCurrentDis + (FluidSurfaceDensity - 0.5) - 1)) ||
				isRealEQ(vioFluidInsideSDFFieldDataGPUPtr[DownLinearIndex], (vCurrentDis + (FluidSurfaceDensity - 0.5) - 1)) ||
				isRealEQ(vioFluidInsideSDFFieldDataGPUPtr[UpLinearIndex], (vCurrentDis + (FluidSurfaceDensity - 0.5) - 1)) ||
				isRealEQ(vioFluidInsideSDFFieldDataGPUPtr[BackLinearIndex], (vCurrentDis + (FluidSurfaceDensity - 0.5) - 1)) ||
				isRealEQ(vioFluidInsideSDFFieldDataGPUPtr[FrontLinearIndex], (vCurrentDis + (FluidSurfaceDensity - 0.5) - 1)))
			{
				vioFluidInsideSDFFieldDataGPUPtr[CenterLinearIndex] = vCurrentDis + (FluidSurfaceDensity - 0.5);
			}
		}
	}
}

__global__ void buildFluidOutsideSDF
(
	Vector3i vResolution,
	const Real* vFluidDensityFieldDataGPUPtr,
	const Real* vSolidDomainFieldDataGPUPtr,
	Real* vioFluidOutsideSDFFieldDataGPUPtr,
	UInt vCurrentDis
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CenterLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);
	Int LeftLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0));
	Int RightLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(1, 0, 0));
	Int DownLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0));
	Int UpLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 1, 0));
	Int BackLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1));
	Int FrontLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, 1));

	if (vCurrentDis == 0)
	{
		if (isInsideSDF(vSolidDomainFieldDataGPUPtr[CenterLinearIndex]))
		{
			vioFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = 0.0;
		}
		else
		{
			if (isRealEQ(vFluidDensityFieldDataGPUPtr[CenterLinearIndex], FluidSurfaceDensity))
			{
				vioFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = (0.5 - FluidSurfaceDensity);
			}
			else
			{
				if (isRealEQ(vFluidDensityFieldDataGPUPtr[CenterLinearIndex], 1.0))
				{
					vioFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = 0.0;
				}
				else
				{
					vioFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = UNKNOWN;
				}
			}
		}
	}
	else
	{
		if (vioFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] == UNKNOWN)
		{
			if (isRealEQ(vioFluidOutsideSDFFieldDataGPUPtr[LeftLinearIndex], (vCurrentDis + (0.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidOutsideSDFFieldDataGPUPtr[RightLinearIndex], (vCurrentDis + (0.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidOutsideSDFFieldDataGPUPtr[DownLinearIndex], (vCurrentDis + (0.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidOutsideSDFFieldDataGPUPtr[UpLinearIndex], (vCurrentDis + (0.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidOutsideSDFFieldDataGPUPtr[BackLinearIndex], (vCurrentDis + (0.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidOutsideSDFFieldDataGPUPtr[FrontLinearIndex], (vCurrentDis + (0.5 - FluidSurfaceDensity) - 1)))
			{
				vioFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = vCurrentDis + (0.5 - FluidSurfaceDensity);
			}
		}
	}
}

__global__ void buildFluidSDFPostProcess
(
	Vector3i vResolution,
	Real* vioFluidSDFFieldDataGPUPtr
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CenterLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);

	if (vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] < 0.0)
	{
		vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] = 0.0;
	}
}

__global__ void buildMixedFluidOutsideSDF
(
	Vector3i vResolution,
	const Real* vGridFluidDensityFieldDataGPUPtr,
	const Real* vMixedFluidDensityFieldDataGPUPtr,
	const Real* vGridFluidOutsideFieldDataGPUPtr,
	Real* vioMixedFluidOutsideSDFFieldDataGPUPtr
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CenterLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);
	Int LeftLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0));
	Int RightLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(1, 0, 0));
	Int DownLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0));
	Int UpLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 1, 0));
	Int BackLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1));
	Int FrontLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, 1));

	if (isRealEQ(vGridFluidDensityFieldDataGPUPtr[CenterLinearIndex], FluidSurfaceDensity))
	{
		vioMixedFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = (0.5 - vMixedFluidDensityFieldDataGPUPtr[CenterLinearIndex]);
		if (vioMixedFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] < 0.0)
		{
			vioMixedFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = 0.0;
		}
	}
	else
	{
		vioMixedFluidOutsideSDFFieldDataGPUPtr[CenterLinearIndex] = vGridFluidOutsideFieldDataGPUPtr[CenterLinearIndex];
	}
}

void buildFluidOutsideSDFInvoker
(
	const CCellCenteredScalarField& vFluidDensityField,
	const CCellCenteredScalarField& vSolidDomainField,
	CCellCenteredScalarField& voFluidOutsideSDFField,
	UInt vExtrapolationDistance
)
{
	Vector3i Resolution = vFluidDensityField.getResolution();

	_ASSERTE(Resolution == vSolidDomainField.getResolution());
	_ASSERTE(Resolution == voFluidOutsideSDFField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	for (UInt i = 0; i < vExtrapolationDistance; i++)
	{
		buildFluidOutsideSDF << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vFluidDensityField.getConstGridDataGPUPtr(),
				vSolidDomainField.getConstGridDataGPUPtr(),
				voFluidOutsideSDFField.getGridDataGPUPtr(),
				i
				);
	}

	buildFluidSDFPostProcess << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			voFluidOutsideSDFField.getGridDataGPUPtr()
		);
}

void buildFluidInsideSDFInvoker
(
	const CCellCenteredScalarField& vFluidDensityField,
	const CCellCenteredScalarField& vSolidDomainField,
	CCellCenteredScalarField& voFluidInsideSDFField,
	UInt vExtrapolationDistance
)
{
	Vector3i Resolution = vFluidDensityField.getResolution();

	_ASSERTE(Resolution == vSolidDomainField.getResolution());
	_ASSERTE(Resolution == voFluidInsideSDFField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	for (UInt i = 0; i < vExtrapolationDistance; i++)
	{
		buildFluidInsideSDF << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vFluidDensityField.getConstGridDataGPUPtr(),
				vSolidDomainField.getConstGridDataGPUPtr(),
				voFluidInsideSDFField.getGridDataGPUPtr(),
				i
			);
	}
	buildFluidSDFPostProcess << <NumBlock, ThreadPerBlock >> >
	(
		Resolution,
		voFluidInsideSDFField.getGridDataGPUPtr()
	);
}

void buildMixedFluidOutsideSDFInvoker
(
	const CCellCenteredScalarField& vGridFluidDensityField,
	const CCellCenteredScalarField& vMixedFluidDensityField,
	const CCellCenteredScalarField& vGridFluidOutsideSDFField,
	CCellCenteredScalarField& voMixedFluidOutsideSDFField
)
{
	Vector3i Resolution = vGridFluidDensityField.getResolution();

	_ASSERTE(Resolution == vMixedFluidDensityField.getResolution());
	_ASSERTE(Resolution == vGridFluidOutsideSDFField.getResolution());
	_ASSERTE(Resolution == voMixedFluidOutsideSDFField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	buildMixedFluidOutsideSDF << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			vGridFluidDensityField.getConstGridDataGPUPtr(),
			vMixedFluidDensityField.getConstGridDataGPUPtr(),
			vGridFluidOutsideSDFField.getConstGridDataGPUPtr(),
			voMixedFluidOutsideSDFField.getGridDataGPUPtr()
		);
}
