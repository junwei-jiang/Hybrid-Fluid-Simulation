#include "EulerParticlesTool.cuh"

__device__ UInt DeviceNumOfFluidGrid;

__global__ void findFluidGrid
(
	Vector3i vResolution, 
	const Real* vFluidSDFDataGPUPtr, 
	const Real* vSolidSDFDataGPUPtr,
	bool* voFluidGridFlagGPUPtr, 
	Int vTotalThreadNum)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	voFluidGridFlagGPUPtr[CurLinearIndex] = false;

	if (isInsideSDF(vFluidSDFDataGPUPtr[CurLinearIndex]) && !isInsideSDF(vSolidSDFDataGPUPtr[CurLinearIndex]))
	{
		atomicAdd(&DeviceNumOfFluidGrid, 1);
		voFluidGridFlagGPUPtr[CurLinearIndex] = true;
	}
}

__global__ void statisticalFluidDensity
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	const Real* vParticlesPosGPUPtr, 
	Real* voMarkersFieldGridDataGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurParticleLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurParticleLinearIndex >= vTotalThreadNum)
		return;

	Vector3 CurParticlesPos = Vector3(vParticlesPosGPUPtr[3 * CurParticleLinearIndex], vParticlesPosGPUPtr[3 * CurParticleLinearIndex + 1], vParticlesPosGPUPtr[3 * CurParticleLinearIndex + 2]);
	Vector3i CorrespondGridIndex = transPos2Index(CurParticlesPos, vResolution, vOrigin, vSpacing);
	Int CorrespondGridLinearIndex = transIndex2LinearWithOffset(CorrespondGridIndex, vResolution);
	atomicAdd(&voMarkersFieldGridDataGPUPtr[CorrespondGridLinearIndex], 1.0);
}

__global__ void buildFluidMarkers
(
	Vector3i vResolution,
	const Real* vSolidSDFDataGPUPtr,
	Real* voMarkersFieldGridDataGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	if (isInsideSDF(vSolidSDFDataGPUPtr[CurLinearIndex]))//固体
	{
		voMarkersFieldGridDataGPUPtr[CurLinearIndex] = 2.0;
	}
	else
	{
		if (voMarkersFieldGridDataGPUPtr[CurLinearIndex] > 0.0)//液体
		{
			voMarkersFieldGridDataGPUPtr[CurLinearIndex] = 1.0;
		}
		else//气体
		{
			voMarkersFieldGridDataGPUPtr[CurLinearIndex] = 0.0;
		}
	}

}

__global__ void accumulateParticles2CCSField
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	const Real* vParticlesPosGPUPtr,
	const Real* vParticlesScalarValueGPUPtr,
	Real* voScalarFieldGridDataGPUPtr,
	Real* voWeightFieldGridDataGPUPtr,
	Int vTotalThreadNum,
	EPGTransferAlgorithm vTransferAlg,
	Int vSrcDataSpan = 1,
	Int vSrcDataOffset = 0
)
{
	Int CurParticleLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurParticleLinearIndex >= vTotalThreadNum)
		return;

	Vector3 CurParticlesPos = Vector3(vParticlesPosGPUPtr[3 * CurParticleLinearIndex], vParticlesPosGPUPtr[3 * CurParticleLinearIndex + 1], vParticlesPosGPUPtr[3 * CurParticleLinearIndex + 2]);
	Vector3 RelPos = (CurParticlesPos - vOrigin - vSpacing * 0.5);
	Vector3 RelPosIndex = RelPos / vSpacing;
	Vector3i DownBackLeftGridIndex = Vector3i(floorCUDA(RelPosIndex.x), floorCUDA(RelPosIndex.y), floorCUDA(RelPosIndex.z));
	Vector3 OffsetVector = (RelPos - Vector3((Real)(DownBackLeftGridIndex.x) * vSpacing.x, (Real)(DownBackLeftGridIndex.y) * vSpacing.y, (Real)(DownBackLeftGridIndex.z) * vSpacing.z)) / vSpacing;
	
	if (vTransferAlg == EPGTransferAlgorithm::P2GSUM)
	{
		Vector3i CorrespondGridIndex = transPos2Index(CurParticlesPos, vResolution, vOrigin, vSpacing);
		Int CorrespondGridLinearIndex = transIndex2LinearWithOffset(CorrespondGridIndex, vResolution);
		atomicAdd(&voScalarFieldGridDataGPUPtr[CorrespondGridLinearIndex], vParticlesScalarValueGPUPtr[vSrcDataSpan * CurParticleLinearIndex + vSrcDataOffset]);
	}
	else if (vTransferAlg == EPGTransferAlgorithm::LINEAR)
	{
		Real WeightX[2] = { linearKernelFunc(OffsetVector.x), linearKernelFunc(OffsetVector.x - 1.0) };
		Real WeightY[2] = { linearKernelFunc(OffsetVector.y), linearKernelFunc(OffsetVector.y - 1.0) };
		Real WeightZ[2] = { linearKernelFunc(OffsetVector.z), linearKernelFunc(OffsetVector.z - 1.0) };

		for (int z = 0; z < 2; z++)
		{
			for (int y = 0; y < 2; y++)
			{
				for (int x = 0; x < 2; x++)
				{
					Real Weight = WeightX[x] * WeightY[y] * WeightZ[z];
					atomicAdd
					(
						&voScalarFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x, y, z))],
						Weight * vParticlesScalarValueGPUPtr[vSrcDataSpan * CurParticleLinearIndex + vSrcDataOffset]
					);
					atomicAdd
					(
						&voWeightFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x, y, z))],
						Weight
					);
				}
			}
		}
	}
	else if (vTransferAlg == EPGTransferAlgorithm::QUADRATIC)
	{
		Real WeightX[4] = { quadraticKernelFunc(OffsetVector.x + 1.0), quadraticKernelFunc(OffsetVector.x), quadraticKernelFunc(OffsetVector.x - 1.0), quadraticKernelFunc(OffsetVector.x - 2.0) };
		Real WeightY[4] = { quadraticKernelFunc(OffsetVector.y + 1.0), quadraticKernelFunc(OffsetVector.y), quadraticKernelFunc(OffsetVector.y - 1.0), quadraticKernelFunc(OffsetVector.y - 2.0) };
		Real WeightZ[4] = { quadraticKernelFunc(OffsetVector.z + 1.0), quadraticKernelFunc(OffsetVector.z), quadraticKernelFunc(OffsetVector.z - 1.0), quadraticKernelFunc(OffsetVector.z - 2.0) };

		for (int z = 0; z < 4; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 4; x++)
				{
					Real Weight = WeightX[x] * WeightY[y] * WeightZ[z];
					atomicAdd
					(
						&voScalarFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x - 1, y - 1, z - 1))],
						Weight * vParticlesScalarValueGPUPtr[vSrcDataSpan * CurParticleLinearIndex + vSrcDataOffset]
					);
					atomicAdd
					(
						&voWeightFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x - 1, y - 1, z - 1))],
						Weight
					);
				}
			}
		}
	}
	else if (vTransferAlg == EPGTransferAlgorithm::CUBIC)
	{
		Real WeightX[4] = { cubicKernelFunc(OffsetVector.x + 1.0), cubicKernelFunc(OffsetVector.x), cubicKernelFunc(OffsetVector.x - 1.0), cubicKernelFunc(OffsetVector.x - 2.0) };
		Real WeightY[4] = { cubicKernelFunc(OffsetVector.y + 1.0), cubicKernelFunc(OffsetVector.y), cubicKernelFunc(OffsetVector.y - 1.0), cubicKernelFunc(OffsetVector.y - 2.0) };
		Real WeightZ[4] = { cubicKernelFunc(OffsetVector.z + 1.0), cubicKernelFunc(OffsetVector.z), cubicKernelFunc(OffsetVector.z - 1.0), cubicKernelFunc(OffsetVector.z - 2.0) };

		for (int z = 0; z < 4; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 4; x++)
				{
					Real Weight = WeightX[x] * WeightY[y] * WeightZ[z];
					atomicAdd
					(
						&voScalarFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x - 1, y - 1, z - 1))],
						Weight * vParticlesScalarValueGPUPtr[vSrcDataSpan * CurParticleLinearIndex + vSrcDataOffset]
					);
					atomicAdd
					(
						&voWeightFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x - 1, y - 1, z - 1))],
						Weight
					);
				}
			}
		}
	}
	else
	{

	}
}

__global__ void accumulateCCSField2Particles
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	const Real* vParticlesPosGPUPtr,
	Real* voParticlesScalarValueGPUPtr,
	const Real* vScalarFieldGridDataGPUPtr,
	Int vTotalThreadNum,
	EPGTransferAlgorithm vTransferAlg,
	Int vDstDataSpan = 1,
	Int vDstDataOffset = 0
)
{
	Int CurParticleLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurParticleLinearIndex >= vTotalThreadNum)
		return;

	Vector3 CurParticlesPos = Vector3(vParticlesPosGPUPtr[3 * CurParticleLinearIndex], vParticlesPosGPUPtr[3 * CurParticleLinearIndex + 1], vParticlesPosGPUPtr[3 * CurParticleLinearIndex + 2]);
	Vector3 RelPos = (CurParticlesPos - vOrigin - vSpacing * 0.5);
	Vector3 RelPosIndex = RelPos / vSpacing;
	Vector3i DownBackLeftGridIndex = Vector3i(floorCUDA(RelPosIndex.x), floorCUDA(RelPosIndex.y), floorCUDA(RelPosIndex.z));
	Vector3 OffsetVector = (RelPos - Vector3((Real)(DownBackLeftGridIndex.x) * vSpacing.x, (Real)(DownBackLeftGridIndex.y) * vSpacing.y, (Real)(DownBackLeftGridIndex.z) * vSpacing.z)) / vSpacing;

	voParticlesScalarValueGPUPtr[vDstDataSpan * CurParticleLinearIndex + vDstDataOffset] = 0.0;
	if (vTransferAlg == EPGTransferAlgorithm::G2PNEAREST)
	{
		Vector3i CorrespondGridIndex = transPos2Index(CurParticlesPos, vResolution, vOrigin, vSpacing);
		Int CorrespondGridLinearIndex = transIndex2LinearWithOffset(CorrespondGridIndex, vResolution);
		voParticlesScalarValueGPUPtr[vDstDataSpan * CurParticleLinearIndex + vDstDataOffset] = vScalarFieldGridDataGPUPtr[CorrespondGridLinearIndex];
	}
	else if (vTransferAlg == EPGTransferAlgorithm::LINEAR)
	{
		Real WeightX[2] = { linearKernelFunc(OffsetVector.x), linearKernelFunc(OffsetVector.x - 1.0) };
		Real WeightY[2] = { linearKernelFunc(OffsetVector.y), linearKernelFunc(OffsetVector.y - 1.0) };
		Real WeightZ[2] = { linearKernelFunc(OffsetVector.z), linearKernelFunc(OffsetVector.z - 1.0) };

		for (int z = 0; z < 2; z++)
		{
			for (int y = 0; y < 2; y++)
			{
				for (int x = 0; x < 2; x++)
				{
					Real Weight = WeightX[x] * WeightY[y] * WeightZ[z];
					voParticlesScalarValueGPUPtr[vDstDataSpan * CurParticleLinearIndex + vDstDataOffset] += Weight * vScalarFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x, y, z))];
				}
			}
		}
	}
	else if (vTransferAlg == EPGTransferAlgorithm::QUADRATIC)
	{
		Real WeightX[4] = { quadraticKernelFunc(OffsetVector.x + 1.0), quadraticKernelFunc(OffsetVector.x), quadraticKernelFunc(OffsetVector.x - 1.0), quadraticKernelFunc(OffsetVector.x - 2.0) };
		Real WeightY[4] = { quadraticKernelFunc(OffsetVector.y + 1.0), quadraticKernelFunc(OffsetVector.y), quadraticKernelFunc(OffsetVector.y - 1.0), quadraticKernelFunc(OffsetVector.y - 2.0) };
		Real WeightZ[4] = { quadraticKernelFunc(OffsetVector.z + 1.0), quadraticKernelFunc(OffsetVector.z), quadraticKernelFunc(OffsetVector.z - 1.0), quadraticKernelFunc(OffsetVector.z - 2.0) };

		for (int z = 0; z < 4; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 4; x++)
				{
					Real Weight = WeightX[x] * WeightY[y] * WeightZ[z];
					voParticlesScalarValueGPUPtr[vDstDataSpan * CurParticleLinearIndex + vDstDataOffset] += Weight * vScalarFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x - 1, y - 1, z - 1))];
				}
			}
		}
	}
	else if (vTransferAlg == EPGTransferAlgorithm::CUBIC)
	{
		Real WeightX[4] = { cubicKernelFunc(OffsetVector.x + 1.0), cubicKernelFunc(OffsetVector.x), cubicKernelFunc(OffsetVector.x - 1.0), cubicKernelFunc(OffsetVector.x - 2.0) };
		Real WeightY[4] = { cubicKernelFunc(OffsetVector.y + 1.0), cubicKernelFunc(OffsetVector.y), cubicKernelFunc(OffsetVector.y - 1.0), cubicKernelFunc(OffsetVector.y - 2.0) };
		Real WeightZ[4] = { cubicKernelFunc(OffsetVector.z + 1.0), cubicKernelFunc(OffsetVector.z), cubicKernelFunc(OffsetVector.z - 1.0), cubicKernelFunc(OffsetVector.z - 2.0) };

		for (int z = 0; z < 4; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 4; x++)
				{
					Real Weight = WeightX[x] * WeightY[y] * WeightZ[z];
					voParticlesScalarValueGPUPtr[vDstDataSpan * CurParticleLinearIndex + vDstDataOffset] += Weight * vScalarFieldGridDataGPUPtr[transIndex2LinearWithOffset(DownBackLeftGridIndex, vResolution, Vector3i(x - 1, y - 1, z - 1))];
				}
			}
		}
	}
	else
	{

	}
}

__global__ void normalizeCCSField
(
	Vector3i vResolution,
	Real* vioScalarFieldDataGPUPtr,
	const Real* vWeightFieldDataGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurLinearIndex >= vTotalThreadNum)
		return;

	if (vWeightFieldDataGPUPtr[CurLinearIndex] > 0.0)
	{
		vioScalarFieldDataGPUPtr[CurLinearIndex] /= vWeightFieldDataGPUPtr[CurLinearIndex];
		//标记为流体Cell
	}
}

__global__ void fixParticlesPosWithBoundarys
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Real* vioParticlesPosGPUPtr,
	const Real* vBoundarysSDFDataGPUPtr,
	Int vTotalThreadNum
)
{
	Int CurParticleLinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (CurParticleLinearIndex >= vTotalThreadNum)
		return;

	Vector3 CurParticlesPos = Vector3(vioParticlesPosGPUPtr[3 * CurParticleLinearIndex], vioParticlesPosGPUPtr[3 * CurParticleLinearIndex + 1], vioParticlesPosGPUPtr[3 * CurParticleLinearIndex + 2]);

	Vector3 Min = vOrigin;
	Vector3 Max = vOrigin + Vector3(vResolution.x * vSpacing.x, vResolution.y * vSpacing.y, vResolution.z * vSpacing.z);

	vioParticlesPosGPUPtr[3 * CurParticleLinearIndex] = clampCUDA(vioParticlesPosGPUPtr[3 * CurParticleLinearIndex], Min.x, Max.x);
	vioParticlesPosGPUPtr[3 * CurParticleLinearIndex + 1] = clampCUDA(vioParticlesPosGPUPtr[3 * CurParticleLinearIndex + 1], Min.y, Max.y);
	vioParticlesPosGPUPtr[3 * CurParticleLinearIndex + 2] = clampCUDA(vioParticlesPosGPUPtr[3 * CurParticleLinearIndex + 2], Min.z, Max.z);
}

void findFluidGridInvoker
(
	const CCellCenteredScalarField& vFluidSDF, 
	const CCellCenteredScalarField& vSolidSDF,
	UInt& voNumOfFluidGrid, 
	thrust::device_vector<bool>& voFluidGridFlag
)
{
	Vector3i Resolution = vFluidSDF.getResolution();
	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	UInt TempNumOfFluidGrid = 0;
	CHECK_CUDA(cudaMemcpyToSymbol(DeviceNumOfFluidGrid, &TempNumOfFluidGrid, sizeof(UInt)));

	findFluidGrid << <NumBlock, ThreadPerBlock >> > 
		(
			Resolution, 
			vFluidSDF.getConstGridDataGPUPtr(), 
			vSolidSDF.getConstGridDataGPUPtr(),
			getRawDevicePointerBool(voFluidGridFlag),
			TotalThreadNum
		);

	CHECK_CUDA(cudaMemcpyFromSymbol(&voNumOfFluidGrid, DeviceNumOfFluidGrid, sizeof(UInt)));
}

void buildFluidMarkersInvoker
(
	const CCellCenteredScalarField& vSolidSDF,
	const thrust::device_vector<Real>& vParticlesPos,
	CCellCenteredScalarField& voMarkersField
)
{
	_ASSERTE(vSolidSDF.getResolution() == voMarkersField.getResolution());

	Vector3i Resolution = vSolidSDF.getResolution();
	Vector3 Origin = vSolidSDF.getOrigin();
	Vector3 Spacing = vSolidSDF.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	voMarkersField.resize(voMarkersField.getResolution(), voMarkersField.getOrigin(), voMarkersField.getSpacing());

	statisticalFluidDensity << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
	(
		Resolution,
		Origin,
		Spacing,
		getReadOnlyRawDevicePointer(vParticlesPos),
		voMarkersField.getGridDataGPUPtr(),
		ParticlesTotalThreadNum
	);

	Int GridTotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(GridTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 GridThreadPerBlock(BlockSize);
	dim3 GridNumBlock(GridSize);

	buildFluidMarkers << <GridNumBlock, GridThreadPerBlock >> >
	(
		Resolution,
		vSolidSDF.getConstGridDataGPUPtr(),
		voMarkersField.getGridDataGPUPtr(),
		GridTotalThreadNum
	);
}

void statisticalFluidDensityInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	CCellCenteredScalarField& voFluidDensityField
)
{
	Vector3i Resolution = voFluidDensityField.getResolution();
	Vector3 Origin = voFluidDensityField.getOrigin();
	Vector3 Spacing = voFluidDensityField.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	voFluidDensityField.resize(voFluidDensityField.getResolution(), voFluidDensityField.getOrigin(), voFluidDensityField.getSpacing());

	statisticalFluidDensity << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			voFluidDensityField.getGridDataGPUPtr(),
			ParticlesTotalThreadNum
		);
}

void tranferParticles2CCSFieldInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	const thrust::device_vector<Real>& vParticlesScalarValue,
	CCellCenteredScalarField& voScalarField,
	CCellCenteredScalarField& voWeightField,
	EPGTransferAlgorithm vTransferAlg,
	bool vIsNormalization
)
{
	_ASSERTE(voScalarField.getResolution() == voWeightField.getResolution());

	Vector3i Resolution = voScalarField.getResolution();
	Vector3 Origin = voScalarField.getOrigin();
	Vector3 Spacing = voScalarField.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	voScalarField.resize(voScalarField.getResolution(), voScalarField.getOrigin(), voScalarField.getSpacing());
	voWeightField.resize(voWeightField.getResolution(), voWeightField.getOrigin(), voWeightField.getSpacing());

	accumulateParticles2CCSField << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getReadOnlyRawDevicePointer(vParticlesScalarValue),
			voScalarField.getGridDataGPUPtr(),
			voWeightField.getGridDataGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg
		);

	if (vIsNormalization)
	{
		Int GridTotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

		CCudaContextManager::getInstance().fetchPropBlockGridSize1D(GridTotalThreadNum, BlockSize, GridSize, 0.125);
		dim3 GridThreadPerBlock(BlockSize);
		dim3 GridNumBlock(GridSize);

		normalizeCCSField << <GridNumBlock, GridThreadPerBlock >> >
			(
				Resolution,
				voScalarField.getGridDataGPUPtr(),
				voWeightField.getConstGridDataGPUPtr(),
				GridTotalThreadNum
			);
	}
}

void tranferParticles2CCVFieldInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	const thrust::device_vector<Real>& vParticlesVectorValue,
	CCellCenteredVectorField& voVectorField,
	CCellCenteredVectorField& voWeightField,
	EPGTransferAlgorithm vTransferAlg,
	bool vIsNormalization
)
{
	_ASSERTE(voVectorField.getResolution() == voWeightField.getResolution());

	Vector3i Resolution = voVectorField.getResolution();
	Vector3 Origin = voVectorField.getOrigin();
	Vector3 Spacing = voVectorField.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	voVectorField.resize(voVectorField.getResolution(), voVectorField.getOrigin(), voVectorField.getSpacing());
	voWeightField.resize(voWeightField.getResolution(), voWeightField.getOrigin(), voWeightField.getSpacing());

	accumulateParticles2CCSField << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getReadOnlyRawDevicePointer(vParticlesVectorValue),
			voVectorField.getGridDataXGPUPtr(),
			voWeightField.getGridDataXGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			0
		);
	accumulateParticles2CCSField << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getReadOnlyRawDevicePointer(vParticlesVectorValue),
			voVectorField.getGridDataYGPUPtr(),
			voWeightField.getGridDataYGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			1
		);
	accumulateParticles2CCSField << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getReadOnlyRawDevicePointer(vParticlesVectorValue),
			voVectorField.getGridDataZGPUPtr(),
			voWeightField.getGridDataZGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			2
		);

	if (vIsNormalization)
	{
		Int GridTotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

		CCudaContextManager::getInstance().fetchPropBlockGridSize1D(GridTotalThreadNum, BlockSize, GridSize, 0.125);
		dim3 GridThreadPerBlock(BlockSize);
		dim3 GridNumBlock(GridSize);

		normalizeCCSField << <GridNumBlock, GridThreadPerBlock >> >
			(
				Resolution,
				voVectorField.getGridDataXGPUPtr(),
				voWeightField.getConstGridDataXGPUPtr(),
				GridTotalThreadNum
			);
		normalizeCCSField << <GridNumBlock, GridThreadPerBlock >> >
			(
				Resolution,
				voVectorField.getGridDataYGPUPtr(),
				voWeightField.getConstGridDataYGPUPtr(),
				GridTotalThreadNum
			);
		normalizeCCSField << <GridNumBlock, GridThreadPerBlock >> >
			(
				Resolution,
				voVectorField.getGridDataZGPUPtr(),
				voWeightField.getConstGridDataZGPUPtr(),
				GridTotalThreadNum
			);
	}
}

void tranferParticles2FCVFieldInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	const thrust::device_vector<Real>& vParticlesVectorValue,
	CFaceCenteredVectorField& voVectorField,
	CFaceCenteredVectorField& voWeightField,
	EPGTransferAlgorithm vTransferAlg,
	bool vIsNormalization
)
{
	Vector3i Resolution = voVectorField.getResolution();
	Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
	Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);
	Vector3  OriginX = voVectorField.getOrigin() - Vector3(voVectorField.getSpacing().x / 2, 0, 0);
	Vector3  OriginY = voVectorField.getOrigin() - Vector3(0, voVectorField.getSpacing().y / 2, 0);
	Vector3  OriginZ = voVectorField.getOrigin() - Vector3(0, 0, voVectorField.getSpacing().z / 2);
	Vector3  Spacing = voVectorField.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	voVectorField.resize(voVectorField.getResolution(), voVectorField.getOrigin(), voVectorField.getSpacing());
	voWeightField.resize(voWeightField.getResolution(), voWeightField.getOrigin(), voWeightField.getSpacing());

	accumulateParticles2CCSField << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			ResolutionX,
			OriginX,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getReadOnlyRawDevicePointer(vParticlesVectorValue),
			voVectorField.getGridDataXGPUPtr(),
			voWeightField.getGridDataXGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3, 
			0
		);
	accumulateParticles2CCSField << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			ResolutionY,
			OriginY,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getReadOnlyRawDevicePointer(vParticlesVectorValue),
			voVectorField.getGridDataYGPUPtr(),
			voWeightField.getGridDataYGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			1
		);
	accumulateParticles2CCSField << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			ResolutionZ,
			OriginZ,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getReadOnlyRawDevicePointer(vParticlesVectorValue),
			voVectorField.getGridDataZGPUPtr(),
			voWeightField.getGridDataZGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			2
		);

	if (vIsNormalization)
	{
		Int GridTotalThreadNumX = (Int)(ResolutionX.x * ResolutionX.y * ResolutionX.z);
		Int GridTotalThreadNumY = (Int)(ResolutionY.x * ResolutionY.y * ResolutionY.z);
		Int GridTotalThreadNumZ = (Int)(ResolutionZ.x * ResolutionZ.y * ResolutionZ.z);

		CCudaContextManager::getInstance().fetchPropBlockGridSize1D(GridTotalThreadNumX, BlockSize, GridSize, 0.125);
		dim3 GridThreadPerBlockX(BlockSize);
		dim3 GridNumBlockX(GridSize);
		CCudaContextManager::getInstance().fetchPropBlockGridSize1D(GridTotalThreadNumY, BlockSize, GridSize, 0.125);
		dim3 GridThreadPerBlockY(BlockSize);
		dim3 GridNumBlockY(GridSize);
		CCudaContextManager::getInstance().fetchPropBlockGridSize1D(GridTotalThreadNumZ, BlockSize, GridSize, 0.125);
		dim3 GridThreadPerBlockZ(BlockSize);
		dim3 GridNumBlockZ(GridSize);

		normalizeCCSField << <GridNumBlockX, GridThreadPerBlockX >> >
			(
				ResolutionX,
				voVectorField.getGridDataXGPUPtr(),
				voWeightField.getConstGridDataXGPUPtr(),
				GridTotalThreadNumX
			);
		normalizeCCSField << <GridNumBlockY, GridThreadPerBlockY >> >
			(
				ResolutionY,
				voVectorField.getGridDataYGPUPtr(),
				voWeightField.getConstGridDataYGPUPtr(),
				GridTotalThreadNumY
			);
		normalizeCCSField << <GridNumBlockZ, GridThreadPerBlockZ >> >
			(
				ResolutionZ,
				voVectorField.getGridDataZGPUPtr(),
				voWeightField.getConstGridDataZGPUPtr(),
				GridTotalThreadNumZ
			);
	}
}

void tranferCCSField2ParticlesInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	thrust::device_vector<Real>& voParticlesScalarValue,
	const CCellCenteredScalarField& vScalarField,
	EPGTransferAlgorithm vTransferAlg
)
{
	Vector3i Resolution = vScalarField.getResolution();
	Vector3 Origin = vScalarField.getOrigin();
	Vector3 Spacing = vScalarField.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	accumulateCCSField2Particles << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getRawDevicePointerReal(voParticlesScalarValue),
			vScalarField.getConstGridDataGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg
		);
}

void tranferCCVField2ParticlesInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	thrust::device_vector<Real>& voParticlesVectorValue,
	const CCellCenteredVectorField& vVectorField,
	EPGTransferAlgorithm vTransferAlg
)
{
	Vector3i Resolution = vVectorField.getResolution();
	Vector3 Origin = vVectorField.getOrigin();
	Vector3 Spacing = vVectorField.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	accumulateCCSField2Particles << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getRawDevicePointerReal(voParticlesVectorValue),
			vVectorField.getConstGridDataXGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			0
		);
	accumulateCCSField2Particles << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getRawDevicePointerReal(voParticlesVectorValue),
			vVectorField.getConstGridDataYGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			1
		);
	accumulateCCSField2Particles << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getRawDevicePointerReal(voParticlesVectorValue),
			vVectorField.getConstGridDataZGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			2
		);
}

void tranferFCVField2ParticlesInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	thrust::device_vector<Real>& voParticlesVectorValue,
	const CFaceCenteredVectorField& vVectorField,
	EPGTransferAlgorithm vTransferAlg
)
{
	Vector3i Resolution = vVectorField.getResolution();
	Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
	Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
	Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);
	Vector3  OriginX = vVectorField.getOrigin() - Vector3(vVectorField.getSpacing().x / 2, 0, 0);
	Vector3  OriginY = vVectorField.getOrigin() - Vector3(0, vVectorField.getSpacing().y / 2, 0);
	Vector3  OriginZ = vVectorField.getOrigin() - Vector3(0, 0, vVectorField.getSpacing().z / 2);
	Vector3  Spacing = vVectorField.getSpacing();

	Int ParticlesTotalThreadNum = (Int)(vParticlesPos.size() / 3);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticlesTotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ParticlesThreadPerBlock(BlockSize);
	dim3 ParticlesNumBlock(GridSize);

	accumulateCCSField2Particles << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			ResolutionX,
			OriginX,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getRawDevicePointerReal(voParticlesVectorValue),
			vVectorField.getConstGridDataXGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3, 
			0
		);
	accumulateCCSField2Particles << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			ResolutionY,
			OriginY,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getRawDevicePointerReal(voParticlesVectorValue),
			vVectorField.getConstGridDataYGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3,
			1
		);
	accumulateCCSField2Particles << <ParticlesNumBlock, ParticlesThreadPerBlock >> >
		(
			ResolutionZ,
			OriginZ,
			Spacing,
			getReadOnlyRawDevicePointer(vParticlesPos),
			getRawDevicePointerReal(voParticlesVectorValue),
			vVectorField.getConstGridDataZGPUPtr(),
			ParticlesTotalThreadNum,
			vTransferAlg,
			3, 
			2
		);
}

void fixParticlesPosWithBoundarysInvoker
(
	thrust::device_vector<Real>& vioParticlesPos,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	Vector3i Resolution = vBoundarysSDF.getResolution();
	Vector3 Origin = vBoundarysSDF.getOrigin();
	Vector3 Spacing = vBoundarysSDF.getSpacing();

	Int TotalThreadNum = (Int)(Resolution.x * Resolution.y * Resolution.z);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(TotalThreadNum, BlockSize, GridSize, 0.125);
	dim3 ThreadPerBlock(BlockSize);
	dim3 NumBlock(GridSize);

	fixParticlesPosWithBoundarys << <NumBlock, ThreadPerBlock >> >
		(
			Resolution,
			Origin,
			Spacing,
			getRawDevicePointerReal(vioParticlesPos),
			vBoundarysSDF.getConstGridDataGPUPtr(),
			TotalThreadNum
		);
}