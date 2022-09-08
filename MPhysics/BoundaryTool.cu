#include "BoundaryTool.cuh"

__global__ void buildSolidsMarker
(
	Vector3i vResolution,
	const Real* vSolidSDFFieldDataGPUPtr,
	Real* voSolidsMarkerFieldDataGPUPtr,
	Int vSolidIndex,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	if (vSolidIndex == 1)
	{
		voSolidsMarkerFieldDataGPUPtr[CurLinearIndex] = UNKNOWN;
	}

	if (isInsideSDF(vSolidSDFFieldDataGPUPtr[CurLinearIndex]) && voSolidsMarkerFieldDataGPUPtr[CurLinearIndex] == UNKNOWN)
	{
		voSolidsMarkerFieldDataGPUPtr[CurLinearIndex] = -vSolidIndex;
	}
}

__global__ void buildSolidsVel
(
	Vector3i vResolution,
	Vector3i vResolutionX,
	Vector3i vResolutionY,
	Vector3i vResolutionZ,
	const Real* vSolidsMarkerFieldDataGPUPtr,
	const Real* vSolidsVel,
	Real* voSolidsVelFieldDataXGPUPtr,
	Real* voSolidsVelFieldDataYGPUPtr,
	Real* voSolidsVelFieldDataZGPUPtr,
	Int vTotalSolidNum,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Vector3i CurCoordIndex = transLinearIndex2Coord(CurLinearIndex, vResolution);

	Int CenterLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolution);
	Int LeftLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolution, Vector3i(-1, 0, 0));
	Int RightLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolution, Vector3i(1, 0, 0));
	Int DownLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolution, Vector3i(0, -1, 0));
	Int UpLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolution, Vector3i(0, 1, 0));
	Int BackLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolution, Vector3i(0, 0, -1));
	Int FrontLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolution, Vector3i(0, 0, 1));

	Int ULeftLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolutionX);
	Int URightLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolutionX, Vector3i(1, 0, 0));
	Int VDownLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolutionY);
	Int VUpLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolutionY, Vector3i(0, 1, 0));
	Int WBackLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolutionZ);
	Int WFrontLinearIndex = transIndex2LinearWithOffset(CurCoordIndex, vResolutionZ, Vector3i(0, 0, 1));

	if (vSolidsMarkerFieldDataGPUPtr[CurLinearIndex] < 0)
	{
		Int CurSolidIndex = -(1 + vSolidsMarkerFieldDataGPUPtr[CurLinearIndex]);

		if (CurSolidIndex >= vTotalSolidNum)
			return;

		atomicExch(&voSolidsVelFieldDataXGPUPtr[ULeftLinearIndex], vSolidsVel[3 * CurSolidIndex]);
		atomicExch(&voSolidsVelFieldDataXGPUPtr[URightLinearIndex], vSolidsVel[3 * CurSolidIndex]);
		atomicExch(&voSolidsVelFieldDataYGPUPtr[VDownLinearIndex], vSolidsVel[3 * CurSolidIndex + 1]);
		atomicExch(&voSolidsVelFieldDataYGPUPtr[VUpLinearIndex], vSolidsVel[3 * CurSolidIndex + 1]);
		atomicExch(&voSolidsVelFieldDataZGPUPtr[WBackLinearIndex], vSolidsVel[3 * CurSolidIndex + 2]);
		atomicExch(&voSolidsVelFieldDataZGPUPtr[WFrontLinearIndex], vSolidsVel[3 * CurSolidIndex + 2]);
	}
	else
	{
		
	}
}

__global__ void transformField
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	const Real* vTranslationGPUPtr,
	const Real* vRotationGPUPtr,
	const Real* vScaleGPUPtr,
	Real* voDstPositionFieldDataXGPUPtr,
	Real* voDstPositionFieldDataYGPUPtr,
	Real* voDstPositionFieldDataZGPUPtr,
	Int vCurSolidIndex,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	Vector3i CurCoordIndex = transLinearIndex2Coord(CurLinearIndex, vResolution);
	Vector3 CurCellPos = vOrigin + 0.5 * vSpacing + Vector3(vSpacing.x * CurCoordIndex.x, vSpacing.y * CurCoordIndex.y, vSpacing.z * CurCoordIndex.z);

	Vector3 DstPos = CurCellPos + Vector3(vTranslationGPUPtr[3 * vCurSolidIndex], vTranslationGPUPtr[3 * vCurSolidIndex + 1], vTranslationGPUPtr[3 * vCurSolidIndex + 2]);

	voDstPositionFieldDataXGPUPtr[CurLinearIndex] = DstPos.x;
	voDstPositionFieldDataYGPUPtr[CurLinearIndex] = DstPos.y;
	voDstPositionFieldDataZGPUPtr[CurLinearIndex] = DstPos.z;
}

void buildSolidsMarkerInvoker(const vector<CCellCenteredScalarField>& vSolidsSDF, CCellCenteredScalarField& voSolidsMarkerField)
{
	Vector3i Resolution = voSolidsMarkerField.getResolution();

	for (int i = 0; i < vSolidsSDF.size(); i++)
	{
		_ASSERTE(Resolution == vSolidsSDF[i].getResolution());
	}

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	voSolidsMarkerField.resize(voSolidsMarkerField.getResolution(), voSolidsMarkerField.getOrigin(), voSolidsMarkerField.getSpacing());

	for (int i = 0; i < vSolidsSDF.size(); i++)
	{
		buildSolidsMarker << <NumBlock, ThreadPerBlock >> >
			(
				Resolution,
				vSolidsSDF[i].getConstGridDataGPUPtr(),
				voSolidsMarkerField.getGridDataGPUPtr(),
				i + 1,
				TotalThreadNum
			);
	}
}

void buildSolidsVelFieldInvoker
(
	const CCellCenteredScalarField& vSolidsMarkerField,
	const thrust::device_vector<Real>& vSolidsVel,
	CFaceCenteredVectorField& voSolidsVelField
)
{
	Vector3i Resolution = vSolidsMarkerField.getResolution();
	Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
	Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);

	_ASSERTE(Resolution == voSolidsVelField.getResolution());
	_ASSERTE(vSolidsVel.size() % 3 == 0);

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	voSolidsVelField.resize(voSolidsVelField.getResolution(), voSolidsVelField.getOrigin(), voSolidsVelField.getSpacing());
	
	buildSolidsVel << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			ResolutionX,
			ResolutionY,
			ResolutionZ,
			vSolidsMarkerField.getConstGridDataGPUPtr(),
			getReadOnlyRawDevicePointer(vSolidsVel),
			voSolidsVelField.getGridDataXGPUPtr(),
			voSolidsVelField.getGridDataYGPUPtr(),
			voSolidsVelField.getGridDataZGPUPtr(),
			vSolidsVel.size() / 3,
			TotalThreadNum
		);
}

void transformFieldInvoker
(
	CCellCenteredVectorField& voDstPostionField,
	const thrust::device_vector<Real>& vTranslation,
	const thrust::device_vector<Real>& vRotation,
	const thrust::device_vector<Real>& vScale,
	Int vCurSolidIndex
)
{
	Vector3i Resolution = voDstPostionField.getResolution();
	Vector3 Origin = voDstPostionField.getOrigin();
	Vector3 Spacing = voDstPostionField.getSpacing();

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	transformField LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vTranslation),
			getReadOnlyRawDevicePointer(vRotation),
			getReadOnlyRawDevicePointer(vScale),
			voDstPostionField.getGridDataXGPUPtr(),
			voDstPostionField.getGridDataYGPUPtr(),
			voDstPostionField.getGridDataZGPUPtr(),
			vCurSolidIndex,
			TotalThreadNum
			);
}