#include "EulerSolverTool.cuh"

__global__ void fixFluidDomain
(
	Vector3i vResolution,
	Real* vioFluidDomainFieldDataGPUPtr,
	const Real* vSolidSDFFieldDataGPUPtr
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CenterLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);

	if (isInsideSDF(vSolidSDFFieldDataGPUPtr[CenterLinearIndex]))//Solid
	{
		vioFluidDomainFieldDataGPUPtr[CenterLinearIndex] = FluidDomainValue;
	}
	else if (isInsideSDF(vioFluidDomainFieldDataGPUPtr[CenterLinearIndex]))//Fluid
	{
		vioFluidDomainFieldDataGPUPtr[CenterLinearIndex] = -FluidDomainValue;
	}
	else//Air
	{
		vioFluidDomainFieldDataGPUPtr[CenterLinearIndex] = FluidDomainValue;
	}
}

__global__ void buildFluidMarkers
(
	Vector3i vResolution,
	const Real* vSolidSDFFieldDataGPUPtr,
	const Real* vFulidSDFFieldDataGPUPtr,
	Real* voMarkersFieldDataGPUPtr
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CurLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);

	if (isInsideSDF(vSolidSDFFieldDataGPUPtr[CurLinearIndex]))//Solid
	{
		voMarkersFieldDataGPUPtr[CurLinearIndex] = 2;
	}
	else if (isInsideSDF(vFulidSDFFieldDataGPUPtr[CurLinearIndex]))//Fluid
	{
		voMarkersFieldDataGPUPtr[CurLinearIndex] = 1;
	}
	else//Air
	{
		voMarkersFieldDataGPUPtr[CurLinearIndex] = 0;
	}
}

__global__ void buildPressureMatrixA
(
	Vector3i vResolution,
	Vector3 vScale,
	const Real* vMarkersFieldDataGPUPtr,
	Real* voFdmMatrixValueGPUPtr
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

	voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] = 0;//Center
	voFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 1] = 0;//Right
	voFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 2] = 0;//Up
	voFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 3] = 0;//Front

	if (vMarkersFieldDataGPUPtr[CenterLinearIndex] == 1)//Fluid
	{
		if (CurIndexX > 0 && vMarkersFieldDataGPUPtr[LeftLinearIndex] != 2)
		{
			voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] += vScale.x;
		}
		if (CurIndexX + 1 < vResolution.x && vMarkersFieldDataGPUPtr[RightLinearIndex] != 2)
		{
			voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] += vScale.x;
			if (vMarkersFieldDataGPUPtr[RightLinearIndex] == 1)
			{
				voFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 1] -= vScale.x;
			}
		}
		if (CurIndexY > 0 && vMarkersFieldDataGPUPtr[DownLinearIndex] != 2)
		{
			voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] += vScale.y;
		}
		if (CurIndexY + 1 < vResolution.y && vMarkersFieldDataGPUPtr[UpLinearIndex] != 2)
		{
			voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] += vScale.y;
			if (vMarkersFieldDataGPUPtr[UpLinearIndex] == 1)
			{
				voFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 2] -= vScale.y;
			}
		}
		if (CurIndexZ > 0 && vMarkersFieldDataGPUPtr[BackLinearIndex] != 2)
		{
			voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] += vScale.z;
		}
		if (CurIndexZ + 1 < vResolution.z && vMarkersFieldDataGPUPtr[FrontLinearIndex] != 2)
		{
			voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] += vScale.z;
			if (vMarkersFieldDataGPUPtr[FrontLinearIndex] == 1)
			{
				voFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 3] -= vScale.z;
			}
		}
	}
	else//Solid & Air
	{
		voFdmMatrixValueGPUPtr[4 * CenterLinearIndex] = 1.0;
	}
}

__global__ void buildPressureVectorb
(
	Vector3i vResolution,
	Vector3i vResolutionX,
	Vector3i vResolutionY,
	Vector3i vResolutionZ,
	Vector3 vScale,
	const Real* vFluidVelFieldDataXGPUPtr,
	const Real* vFluidVelFieldDataYGPUPtr,
	const Real* vFluidVelFieldDataZGPUPtr,
	const Real* vDivergenceFieldDataGPUPtr,
	const Real* vMarkersFieldDataGPUPtr,
	const Real* vSolidVelFieldDataXGPUPtr,
	const Real* vSolidVelFieldDataYGPUPtr,
	const Real* vSolidVelFieldDataZGPUPtr,
	Real* voVectorbValueGPUPtr
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

	Int ULeftLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX);
	Int URightLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(1, 0, 0));
	Int VDownLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY);
	Int VUpLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 1, 0));
	Int WBackLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ);
	Int WFrontLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0, 0, 1));

	if (vMarkersFieldDataGPUPtr[CenterLinearIndex] == 1)
	{
		voVectorbValueGPUPtr[CenterLinearIndex] = -vDivergenceFieldDataGPUPtr[CenterLinearIndex];
		if (vMarkersFieldDataGPUPtr[LeftLinearIndex] == 2)
		{
			voVectorbValueGPUPtr[CenterLinearIndex] -= vScale.x * (vFluidVelFieldDataXGPUPtr[ULeftLinearIndex] - vSolidVelFieldDataXGPUPtr[ULeftLinearIndex]);
		}
		if (vMarkersFieldDataGPUPtr[RightLinearIndex] == 2)
		{
			voVectorbValueGPUPtr[CenterLinearIndex] += vScale.x * (vFluidVelFieldDataXGPUPtr[URightLinearIndex] - vSolidVelFieldDataXGPUPtr[URightLinearIndex]);
		}
		if (vMarkersFieldDataGPUPtr[DownLinearIndex] == 2)
		{
			voVectorbValueGPUPtr[CenterLinearIndex] -= vScale.y * (vFluidVelFieldDataYGPUPtr[VDownLinearIndex] - vSolidVelFieldDataYGPUPtr[VDownLinearIndex]);
		}
		if (vMarkersFieldDataGPUPtr[UpLinearIndex] == 2)
		{
			voVectorbValueGPUPtr[CenterLinearIndex] += vScale.y * (vFluidVelFieldDataYGPUPtr[VUpLinearIndex] - vSolidVelFieldDataYGPUPtr[VUpLinearIndex]);
		}
		if (vMarkersFieldDataGPUPtr[BackLinearIndex] == 2)
		{
			voVectorbValueGPUPtr[CenterLinearIndex] -= vScale.z * (vFluidVelFieldDataZGPUPtr[WBackLinearIndex] - vSolidVelFieldDataZGPUPtr[WBackLinearIndex]);
		}
		if (vMarkersFieldDataGPUPtr[FrontLinearIndex] == 2)
		{
			voVectorbValueGPUPtr[CenterLinearIndex] += vScale.z * (vFluidVelFieldDataZGPUPtr[WFrontLinearIndex] - vSolidVelFieldDataZGPUPtr[WFrontLinearIndex]);
		}
	}
	else
	{
		voVectorbValueGPUPtr[CenterLinearIndex] = 0.0;
	}
}

__global__ void fdmMatrixVectorMul
(
	Vector3i vResolution,
	const Real* vFdmMatrixValueGPUPtr,
	const Real* vInputVectorValueGPUPtr,
	Real* voOutputVectorValueGPUPtr
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

	voOutputVectorValueGPUPtr[CenterLinearIndex] =
		vFdmMatrixValueGPUPtr[4 * CenterLinearIndex] * vInputVectorValueGPUPtr[CenterLinearIndex] +
		((CurIndexX > 0) ? vFdmMatrixValueGPUPtr[4 * LeftLinearIndex + 1] * vInputVectorValueGPUPtr[LeftLinearIndex] : 0.0) +
		((CurIndexX + 1 < vResolution.x) ? vFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 1] * vInputVectorValueGPUPtr[RightLinearIndex] : 0.0) +
		((CurIndexY > 0) ? vFdmMatrixValueGPUPtr[4 * DownLinearIndex + 2] * vInputVectorValueGPUPtr[DownLinearIndex] : 0.0) +
		((CurIndexY + 1 < vResolution.y) ? vFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 2] * vInputVectorValueGPUPtr[UpLinearIndex] : 0.0) +
		((CurIndexZ > 0) ? vFdmMatrixValueGPUPtr[4 * BackLinearIndex + 3] * vInputVectorValueGPUPtr[BackLinearIndex] : 0.0) +
		((CurIndexZ + 1 < vResolution.z) ? vFdmMatrixValueGPUPtr[4 * CenterLinearIndex + 3] * vInputVectorValueGPUPtr[FrontLinearIndex] : 0.0);
}

__global__ void applyPressureGradient
(
	Vector3i vResolution,
	Vector3i vResolutionX,
	Vector3i vResolutionY,
	Vector3i vResolutionZ,
	Vector3 vScale,
	const Real* vMarkersFieldDataGPUPtr,
	const Real* vPressureFieldDataGPUPtr,
	Real* vioFluidVelFieldDataXGPUPtr,
	Real* vioFluidVelFieldDataYGPUPtr,
	Real* vioFluidVelFieldDataZGPUPtr,
	const Real* vSolidVelFieldDataXGPUPtr,
	const Real* vSolidVelFieldDataYGPUPtr,
	const Real* vSolidVelFieldDataZGPUPtr
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CenterLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);
	Int LeftLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(-1, 0, 0));
	Int DownLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, -1, 0));
	Int BackLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution, Vector3i(0, 0, -1));

	Int ULeftLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX);
	Int VDownLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY);
	Int WBackLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ);

	if (vMarkersFieldDataGPUPtr[LeftLinearIndex] == 1 || vMarkersFieldDataGPUPtr[CenterLinearIndex] == 1)
	{
		if (vMarkersFieldDataGPUPtr[LeftLinearIndex] == 2 || vMarkersFieldDataGPUPtr[CenterLinearIndex] == 2)
		{
			vioFluidVelFieldDataXGPUPtr[ULeftLinearIndex] = vSolidVelFieldDataXGPUPtr[ULeftLinearIndex];
		}
		else
		{
			vioFluidVelFieldDataXGPUPtr[ULeftLinearIndex] -= vScale.x * (vPressureFieldDataGPUPtr[CenterLinearIndex] - vPressureFieldDataGPUPtr[LeftLinearIndex]);
		}
	}
	else
	{
		vioFluidVelFieldDataXGPUPtr[ULeftLinearIndex] = UNKNOWN;
	}

	if (vMarkersFieldDataGPUPtr[DownLinearIndex] == 1 || vMarkersFieldDataGPUPtr[CenterLinearIndex] == 1)
	{
		if (vMarkersFieldDataGPUPtr[DownLinearIndex] == 2 || vMarkersFieldDataGPUPtr[CenterLinearIndex] == 2)
		{
			vioFluidVelFieldDataYGPUPtr[VDownLinearIndex] = vSolidVelFieldDataYGPUPtr[VDownLinearIndex];
		}
		else
		{
			vioFluidVelFieldDataYGPUPtr[VDownLinearIndex] -= vScale.y * (vPressureFieldDataGPUPtr[CenterLinearIndex] - vPressureFieldDataGPUPtr[DownLinearIndex]);
		}
	}
	else
	{
		vioFluidVelFieldDataYGPUPtr[VDownLinearIndex] = UNKNOWN;
	}

	if (vMarkersFieldDataGPUPtr[BackLinearIndex] == 1 || vMarkersFieldDataGPUPtr[CenterLinearIndex] == 1)
	{
		if (vMarkersFieldDataGPUPtr[BackLinearIndex] == 2 || vMarkersFieldDataGPUPtr[CenterLinearIndex] == 2)
		{
			vioFluidVelFieldDataZGPUPtr[WBackLinearIndex] = vSolidVelFieldDataZGPUPtr[WBackLinearIndex];
		}
		else
		{
			vioFluidVelFieldDataZGPUPtr[WBackLinearIndex] -= vScale.z * (vPressureFieldDataGPUPtr[CenterLinearIndex] - vPressureFieldDataGPUPtr[BackLinearIndex]);
		}
	}
	else
	{
		vioFluidVelFieldDataZGPUPtr[WBackLinearIndex] = UNKNOWN;
	}

	if (CurIndexX == vResolution.x - 1)
	{
		Int URightLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionX, Vector3i(1, 0, 0));
		vioFluidVelFieldDataXGPUPtr[URightLinearIndex] = UNKNOWN;
	}
	if (CurIndexY == vResolution.y - 1)
	{
		Int VUpLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionY, Vector3i(0, 1, 0));
		vioFluidVelFieldDataYGPUPtr[VUpLinearIndex] = UNKNOWN;
	}
	if (CurIndexZ == vResolution.z - 1)
	{
		Int WFrontLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolutionZ, Vector3i(0, 0, 1));
		vioFluidVelFieldDataZGPUPtr[WFrontLinearIndex] = UNKNOWN;
	}
}

__global__ void buildFluidSDF
(
	Vector3i vResolution,
	const Real* vFluidDomainFieldDataGPUPtr,
	Real* vioFluidSDFFieldDataGPUPtr,
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
		if (isInsideSDF(vFluidDomainFieldDataGPUPtr[CenterLinearIndex]))
		{
			if (!isInsideSDF(vFluidDomainFieldDataGPUPtr[LeftLinearIndex]) ||
				!isInsideSDF(vFluidDomainFieldDataGPUPtr[RightLinearIndex]) ||
				!isInsideSDF(vFluidDomainFieldDataGPUPtr[DownLinearIndex]) ||
				!isInsideSDF(vFluidDomainFieldDataGPUPtr[UpLinearIndex]) ||
				!isInsideSDF(vFluidDomainFieldDataGPUPtr[BackLinearIndex]) ||
				!isInsideSDF(vFluidDomainFieldDataGPUPtr[FrontLinearIndex]))
			{
				vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] = (0.5 - FluidSurfaceDensity);
			}
			else
			{
				vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] = -UNKNOWN;
			}
		}
		else
		{
			if (isInsideSDF(vFluidDomainFieldDataGPUPtr[LeftLinearIndex]) ||
				isInsideSDF(vFluidDomainFieldDataGPUPtr[RightLinearIndex]) ||
				isInsideSDF(vFluidDomainFieldDataGPUPtr[DownLinearIndex]) ||
				isInsideSDF(vFluidDomainFieldDataGPUPtr[UpLinearIndex]) ||
				isInsideSDF(vFluidDomainFieldDataGPUPtr[BackLinearIndex]) ||
				isInsideSDF(vFluidDomainFieldDataGPUPtr[FrontLinearIndex]))
			{
				vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] = (1.5 - FluidSurfaceDensity);
			}
			else
			{
				vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] = UNKNOWN;
			}
		}
	}
	else
	{
		if (vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] == UNKNOWN)
		{
			if (isRealEQ(vioFluidSDFFieldDataGPUPtr[LeftLinearIndex], (vCurrentDis + (1.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[RightLinearIndex], (vCurrentDis + (1.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[DownLinearIndex], (vCurrentDis + (1.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[UpLinearIndex], (vCurrentDis + (1.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[BackLinearIndex], (vCurrentDis + (1.5 - FluidSurfaceDensity) - 1)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[FrontLinearIndex], (vCurrentDis + (1.5 - FluidSurfaceDensity) - 1)))
			{
				vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] = vCurrentDis + (1.5 - FluidSurfaceDensity);
			}
		}
		else if (vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] == -UNKNOWN)
		{
			if (isRealEQ(vioFluidSDFFieldDataGPUPtr[LeftLinearIndex], -(vCurrentDis + FluidSurfaceDensity - 1.5)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[RightLinearIndex], -(vCurrentDis + FluidSurfaceDensity - 1.5)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[DownLinearIndex], -(vCurrentDis + FluidSurfaceDensity - 1.5)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[UpLinearIndex], -(vCurrentDis + FluidSurfaceDensity - 1.5)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[BackLinearIndex], -(vCurrentDis + FluidSurfaceDensity - 1.5)) ||
				isRealEQ(vioFluidSDFFieldDataGPUPtr[FrontLinearIndex], -(vCurrentDis + FluidSurfaceDensity - 1.5)))
			{
				vioFluidSDFFieldDataGPUPtr[CenterLinearIndex] = -(vCurrentDis - (0.5 - FluidSurfaceDensity));
			}
		}
	}
}

__global__ void buildFluidDensity
(
	Vector3i vResolution,
	const Real* vFluidDomainFieldDataGPUPtr,
	const Real* vSolidDomainFieldDataGPUPtr,
	Real* vioFluidDensityFieldDataGPUPtr
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

	if (isInsideSDF(vFluidDomainFieldDataGPUPtr[CenterLinearIndex]) && !isInsideSDF(vSolidDomainFieldDataGPUPtr[CenterLinearIndex]))
	{
		if (isInsideAir(vFluidDomainFieldDataGPUPtr[LeftLinearIndex], vSolidDomainFieldDataGPUPtr[LeftLinearIndex]) ||
			isInsideAir(vFluidDomainFieldDataGPUPtr[RightLinearIndex], vSolidDomainFieldDataGPUPtr[RightLinearIndex]) ||
			isInsideAir(vFluidDomainFieldDataGPUPtr[DownLinearIndex], vSolidDomainFieldDataGPUPtr[DownLinearIndex]) ||
			isInsideAir(vFluidDomainFieldDataGPUPtr[UpLinearIndex], vSolidDomainFieldDataGPUPtr[UpLinearIndex]) ||
			isInsideAir(vFluidDomainFieldDataGPUPtr[BackLinearIndex], vSolidDomainFieldDataGPUPtr[BackLinearIndex]) ||
			isInsideAir(vFluidDomainFieldDataGPUPtr[FrontLinearIndex], vSolidDomainFieldDataGPUPtr[FrontLinearIndex]))
		{
			vioFluidDensityFieldDataGPUPtr[CenterLinearIndex] = FluidSurfaceDensity;
		}
		else
		{
			vioFluidDensityFieldDataGPUPtr[CenterLinearIndex] = 1.0;
		}
	}
	else
	{
		vioFluidDensityFieldDataGPUPtr[CenterLinearIndex] = 0.0;
	}
}

__global__ void buildExtrapolationMarkers
(
	Vector3i vResolution,
	const Real* vScalarFieldDataGPUPtr,
	Real* vioDisMarkersFieldDataGPUPtr,
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
		if (vScalarFieldDataGPUPtr[CenterLinearIndex] == UNKNOWN)
		{
			vioDisMarkersFieldDataGPUPtr[CenterLinearIndex] = UNKNOWN;
		}
		else
		{
			vioDisMarkersFieldDataGPUPtr[CenterLinearIndex] = 0;
		}
	}
	else
	{
		if (
			(vioDisMarkersFieldDataGPUPtr[CenterLinearIndex] == UNKNOWN) &&
			(vioDisMarkersFieldDataGPUPtr[LeftLinearIndex] == vCurrentDis - 1 ||
				vioDisMarkersFieldDataGPUPtr[RightLinearIndex] == vCurrentDis - 1 ||
				vioDisMarkersFieldDataGPUPtr[DownLinearIndex] == vCurrentDis - 1 ||
				vioDisMarkersFieldDataGPUPtr[UpLinearIndex] == vCurrentDis - 1 ||
				vioDisMarkersFieldDataGPUPtr[BackLinearIndex] == vCurrentDis - 1 ||
				vioDisMarkersFieldDataGPUPtr[FrontLinearIndex] == vCurrentDis - 1))
		{
			vioDisMarkersFieldDataGPUPtr[CenterLinearIndex] = vCurrentDis;
		}
	}
}

__global__ void extrapolatingData
(
	Vector3i vResolution,
	Real* vioScalarFieldDataGPUPtr,
	const Real* vDisMarkersFieldDataGPUPtr,
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

	if (vCurrentDis != 0 && vDisMarkersFieldDataGPUPtr[CenterLinearIndex] == vCurrentDis)
	{
		Real NumOfValidNeighbor = 0.0;
		vioScalarFieldDataGPUPtr[CenterLinearIndex] = 0.0;
		if (vDisMarkersFieldDataGPUPtr[LeftLinearIndex] < vCurrentDis)
		{
			vioScalarFieldDataGPUPtr[CenterLinearIndex] += vioScalarFieldDataGPUPtr[LeftLinearIndex];
			NumOfValidNeighbor += 1.0;
		}
		if (vDisMarkersFieldDataGPUPtr[RightLinearIndex] < vCurrentDis)
		{
			vioScalarFieldDataGPUPtr[CenterLinearIndex] += vioScalarFieldDataGPUPtr[RightLinearIndex];
			NumOfValidNeighbor += 1.0;
		}
		if (vDisMarkersFieldDataGPUPtr[DownLinearIndex] < vCurrentDis)
		{
			vioScalarFieldDataGPUPtr[CenterLinearIndex] += vioScalarFieldDataGPUPtr[DownLinearIndex];
			NumOfValidNeighbor += 1.0;
		}
		if (vDisMarkersFieldDataGPUPtr[UpLinearIndex] < vCurrentDis)
		{
			vioScalarFieldDataGPUPtr[CenterLinearIndex] += vioScalarFieldDataGPUPtr[UpLinearIndex];
			NumOfValidNeighbor += 1.0;
		}
		if (vDisMarkersFieldDataGPUPtr[BackLinearIndex] < vCurrentDis)
		{
			vioScalarFieldDataGPUPtr[CenterLinearIndex] += vioScalarFieldDataGPUPtr[BackLinearIndex];
			NumOfValidNeighbor += 1.0;
		}
		if (vDisMarkersFieldDataGPUPtr[FrontLinearIndex] < vCurrentDis)
		{
			vioScalarFieldDataGPUPtr[CenterLinearIndex] += vioScalarFieldDataGPUPtr[FrontLinearIndex];
			NumOfValidNeighbor += 1.0;
		}

		if (NumOfValidNeighbor != 0.0)
		{
			vioScalarFieldDataGPUPtr[CenterLinearIndex] /= NumOfValidNeighbor;
		}
	}
}

__global__ void fillSDFField
(
	Vector3i vResolution,
	cudaTextureObject_t vSDFTexture,
	Real* voSDFFieldDataGPUPtr,
	bool vIsInvSign
)
{
	Int CurIndexX = threadIdx.x;
	Int CurIndexY = blockIdx.x;
	Int CurIndexZ = blockIdx.y;

	Int CurLinearIndex = transIndex2LinearWithOffset(Vector3i(CurIndexX, CurIndexY, CurIndexZ), vResolution);

	Real CurSDFValue = (Real)tex3D<float>(vSDFTexture, CurIndexX, CurIndexY, CurIndexZ);
	if (vIsInvSign)
	{
		CurSDFValue *= -1;
	}
	voSDFFieldDataGPUPtr[CurLinearIndex] = CurSDFValue;
}

__global__ void generateFluidDomainFromBBox
(
	Vector3 vMin,
	Vector3 vMax,
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Real* vFluidDomainGridDataGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Vector3i CurCoordIndex = transLinearIndex2Coord(CurLinearIndex, vResolution);
	Vector3 CurCellPos = vOrigin + 0.5 * vSpacing + Vector3(vSpacing.x * CurCoordIndex.x, vSpacing.y * CurCoordIndex.y, vSpacing.z * CurCoordIndex.z);

	if (CurCellPos.x > vMin.x && CurCellPos.y > vMin.y && CurCellPos.z > vMin.z && CurCellPos.x < vMax.x && CurCellPos.y < vMax.y && CurCellPos.z < vMax.z)
	{
		vFluidDomainGridDataGPUPtr[CurLinearIndex] = -FluidDomainValue;
	}
	else
	{
		vFluidDomainGridDataGPUPtr[CurLinearIndex] = FluidDomainValue;
	}
}

void fixFluidDomainInvoker(const CCellCenteredScalarField& vSolidSDF, CCellCenteredScalarField& vioFluidDomain)
{
	Vector3i Resolution = vSolidSDF.getResolution();

	_ASSERTE(vioFluidDomain.getResolution() == Resolution);

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	fixFluidDomain << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			vioFluidDomain.getGridDataGPUPtr(),
			vSolidSDF.getConstGridDataGPUPtr()
		);
}

void buildFluidMarkersInvoker(const CCellCenteredScalarField& vSolidSDF, const CCellCenteredScalarField& vFluidSDF, CCellCenteredScalarField& voMarkersField)
{
	Vector3i Resolution = voMarkersField.getResolution();

	_ASSERTE(vSolidSDF.getResolution() == Resolution);
	_ASSERTE(vFluidSDF.getResolution() == Resolution);

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	buildFluidMarkers << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			vSolidSDF.getConstGridDataGPUPtr(),
			vFluidSDF.getConstGridDataGPUPtr(),
			voMarkersField.getGridDataGPUPtr()
		);
}

void buildPressureFdmMatrixAInvoker(Vector3i vResolution, Vector3 vScale, const CCellCenteredScalarField& voMarkersField, thrust::device_vector<Real>& voFdmMatrix)
{
	_ASSERTE(vResolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(vResolution.x);
	dim3 NumBlock(vResolution.y, vResolution.z);

	buildPressureMatrixA << <NumBlock, ThreadPerBlock >> >
		(
			vResolution,
			vScale,
			voMarkersField.getConstGridDataGPUPtr(),
			getRawDevicePointerReal(voFdmMatrix)
		);
}

void buildPressureVectorbInvoker
(
	const CFaceCenteredVectorField& vFluidVelField,
	const CCellCenteredScalarField& vVelDivergenceField,
	const CCellCenteredScalarField& vMarkersField,
	const CFaceCenteredVectorField& vSolidVelField,
	CCuDenseVector& voVectorb
)
{
	Vector3i Resolution = vFluidVelField.getResolution();
	Vector3i ResolutionX = vFluidVelField.getResolution() + Vector3i(1, 0, 0);
	Vector3i ResolutionY = vFluidVelField.getResolution() + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = vFluidVelField.getResolution() + Vector3i(0, 0, 1);

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	Vector3 Scale = 1.0 / vFluidVelField.getSpacing();

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	buildPressureVectorb << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			ResolutionX,
			ResolutionY,
			ResolutionZ,
			Scale,
			vFluidVelField.getConstGridDataXGPUPtr(),
			vFluidVelField.getConstGridDataYGPUPtr(),
			vFluidVelField.getConstGridDataZGPUPtr(),
			vVelDivergenceField.getConstGridDataGPUPtr(),
			vMarkersField.getConstGridDataGPUPtr(),
			vSolidVelField.getConstGridDataXGPUPtr(),
			vSolidVelField.getConstGridDataYGPUPtr(),
			vSolidVelField.getConstGridDataZGPUPtr(),
			voVectorb.getVectorValueGPUPtr()
		);
}

void fdmMatrixVectorMulInvoker(Vector3i vResolution, const thrust::device_vector<Real>& vFdmMatrix, const CCuDenseVector& vInputVector, CCuDenseVector& voOutputVector)
{
	_ASSERTE(vResolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(vResolution.x);
	dim3 NumBlock(vResolution.y, vResolution.z);

	fdmMatrixVectorMul << <NumBlock, ThreadPerBlock >> >
		(
			vResolution,
			getReadOnlyRawDevicePointer(vFdmMatrix),
			vInputVector.getConstVectorValueGPUPtr(),
			voOutputVector.getVectorValueGPUPtr()
		);
}

void applyPressureGradientInvoker
(
	Vector3i vResolution,
	Vector3 vScale,
	const CCellCenteredScalarField& vMarkersField,
	const Real* vPressureFieldDataGPUPtr,
	CFaceCenteredVectorField& vioFluidVelField,
	const CFaceCenteredVectorField& vSolidVelField
)
{
	Vector3i Resolution = vioFluidVelField.getResolution();
	Vector3i ResolutionX = vioFluidVelField.getResolution() + Vector3i(1, 0, 0);
	Vector3i ResolutionY = vioFluidVelField.getResolution() + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = vioFluidVelField.getResolution() + Vector3i(0, 0, 1);

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	applyPressureGradient << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			ResolutionX,
			ResolutionY,
			ResolutionZ,
			vScale,
			vMarkersField.getConstGridDataGPUPtr(),
			vPressureFieldDataGPUPtr,
			vioFluidVelField.getGridDataXGPUPtr(),
			vioFluidVelField.getGridDataYGPUPtr(),
			vioFluidVelField.getGridDataZGPUPtr(),
			vSolidVelField.getConstGridDataXGPUPtr(),
			vSolidVelField.getConstGridDataYGPUPtr(),
			vSolidVelField.getConstGridDataZGPUPtr()
		);
}

void buildFluidSDFInvoker
(
	const CCellCenteredScalarField& vFluidDomainField,
	CCellCenteredScalarField& voFluidSDFField,
	UInt vExtrapolationDistance
)
{
	Vector3i Resolution = vFluidDomainField.getResolution();

	_ASSERTE(Resolution == voFluidSDFField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	for (UInt i = 0; i < vExtrapolationDistance; i++)
	{
		buildFluidSDF << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vFluidDomainField.getConstGridDataGPUPtr(),
				voFluidSDFField.getGridDataGPUPtr(),
				i
			);
	}
}

void buildFluidDensityInvoker
(
	const CCellCenteredScalarField& vFluidDomainField,
	const CCellCenteredScalarField& vSolidDomainField,
	CCellCenteredScalarField& voFluidDensityField
)
{
	Vector3i Resolution = vFluidDomainField.getResolution();

	_ASSERTE(Resolution == vSolidDomainField.getResolution());
	_ASSERTE(Resolution == voFluidDensityField.getResolution());
	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	buildFluidDensity << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			vFluidDomainField.getConstGridDataGPUPtr(),
			vSolidDomainField.getConstGridDataGPUPtr(),
			voFluidDensityField.getGridDataGPUPtr()
		);
}

void extrapolatingDataInvoker(CCellCenteredScalarField& vioScalarField, CCellCenteredScalarField& voDisMarkersField, UInt vExtrapolationDistance)
{
	Vector3i Resolution = vioScalarField.getResolution();

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	for (UInt i = 0; i < vExtrapolationDistance; i++)
	{
		buildExtrapolationMarkers << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioScalarField.getConstGridDataGPUPtr(),
				voDisMarkersField.getGridDataGPUPtr(),
				i
			);
	}

	for (UInt i = 1; i < vExtrapolationDistance; i++)
	{
		extrapolatingData << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioScalarField.getGridDataGPUPtr(),
				voDisMarkersField.getConstGridDataGPUPtr(),
				i
			);
	}
}

void extrapolatingDataInvoker(CCellCenteredVectorField& vioVectorField, CCellCenteredVectorField& voDisMarkersField, UInt vExtrapolationDistance)
{
	Vector3i Resolution = vioVectorField.getResolution();

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	for (UInt i = 0; i < vExtrapolationDistance; i++)
	{
		buildExtrapolationMarkers << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioVectorField.getConstGridDataXGPUPtr(),
				voDisMarkersField.getGridDataXGPUPtr(),
				i
			);
		buildExtrapolationMarkers << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioVectorField.getConstGridDataYGPUPtr(),
				voDisMarkersField.getGridDataYGPUPtr(),
				i
			);
		buildExtrapolationMarkers << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioVectorField.getConstGridDataZGPUPtr(),
				voDisMarkersField.getGridDataZGPUPtr(),
				i
			);
	}

	for (UInt i = 1; i < vExtrapolationDistance; i++)
	{
		extrapolatingData << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioVectorField.getGridDataXGPUPtr(),
				voDisMarkersField.getConstGridDataXGPUPtr(),
				i
			);
		extrapolatingData << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioVectorField.getGridDataYGPUPtr(),
				voDisMarkersField.getConstGridDataYGPUPtr(),
				i
			);
		extrapolatingData << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vioVectorField.getGridDataZGPUPtr(),
				voDisMarkersField.getConstGridDataZGPUPtr(),
				i
			);
	}
}

void extrapolatingDataInvoker(CFaceCenteredVectorField& vioVectorField, CFaceCenteredVectorField& voDisMarkersField, UInt vExtrapolationDistance)
{
	Vector3i Resolution = vioVectorField.getResolution();
	Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
	Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlockX(ResolutionX.x);
	dim3 NumBlockX(ResolutionX.y, ResolutionX.z);
	dim3 ThreadPerBlockY(ResolutionY.x);
	dim3 NumBlockY(ResolutionY.y, ResolutionY.z);
	dim3 ThreadPerBlockZ(ResolutionZ.x);
	dim3 NumBlockZ(ResolutionZ.y, ResolutionZ.z);

	for (UInt i = 0; i < vExtrapolationDistance; i++)
	{
		buildExtrapolationMarkers << <NumBlockX, ThreadPerBlockX >> >
			(
				ResolutionX,
				vioVectorField.getConstGridDataXGPUPtr(),
				voDisMarkersField.getGridDataXGPUPtr(),
				i
			);
		buildExtrapolationMarkers << <NumBlockY, ThreadPerBlockY >> >
			(
				ResolutionY,
				vioVectorField.getConstGridDataYGPUPtr(),
				voDisMarkersField.getGridDataYGPUPtr(),
				i
			);
		buildExtrapolationMarkers << <NumBlockZ, ThreadPerBlockZ >> >
			(
				ResolutionZ,
				vioVectorField.getConstGridDataZGPUPtr(),
				voDisMarkersField.getGridDataZGPUPtr(),
				i
			);
	}

	for (UInt i = 1; i < vExtrapolationDistance; i++)
	{
		extrapolatingData << <NumBlockX, ThreadPerBlockX >> >
			(
				ResolutionX,
				vioVectorField.getGridDataXGPUPtr(),
				voDisMarkersField.getConstGridDataXGPUPtr(),
				i
			);
		extrapolatingData << <NumBlockY, ThreadPerBlockY >> >
			(
				ResolutionY,
				vioVectorField.getGridDataYGPUPtr(),
				voDisMarkersField.getConstGridDataYGPUPtr(),
				i
			);
		extrapolatingData << <NumBlockZ, ThreadPerBlockZ >> >
			(
				ResolutionZ,
				vioVectorField.getGridDataZGPUPtr(),
				voDisMarkersField.getConstGridDataZGPUPtr(),
				i
			);
	}
}

void fillSDFFieldInvoker(cudaTextureObject_t vSDFTexture, CCellCenteredScalarField& voSDFField, bool vIsInvSign)
{
	Vector3i Resolution = voSDFField.getResolution();

	_ASSERTE(Resolution.x <= CCudaContextManager::getInstance().getMaxThreadNumberEachBlock());

	dim3 ThreadPerBlock(Resolution.x);
	dim3 NumBlock(Resolution.y, Resolution.z);

	fillSDFField << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			vSDFTexture,
			voSDFField.getGridDataGPUPtr(),
			vIsInvSign
		);
}

void generateFluidDomainFromBBoxInvoker(Vector3 vMin, Vector3 vMax, CCellCenteredScalarField& voFluidDomainField)
{
	Vector3i Resolution = voFluidDomainField.getResolution();
	Vector3 Origin = voFluidDomainField.getOrigin();
	Vector3 Spacing = voFluidDomainField.getSpacing();

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	generateFluidDomainFromBBox LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
		(
			vMin,
			vMax,
			Resolution,
			Origin,
			Spacing,
			voFluidDomainField.getGridDataGPUPtr(),
			TotalThreadNum
		);
}