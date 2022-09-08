#include "EulerParticles.h"
#include "ThrustWapper.cuh"
#include "EulerParticlesTool.cuh"

CEulerParticles::CEulerParticles()
{

}

CEulerParticles::CEulerParticles(UInt vNumOfParticles, const Real* vParticlesPos, const Real* vParticlesVel, const Real* vParticlesColor)
{
	resize(vNumOfParticles, vParticlesPos, vParticlesVel, vParticlesColor);
}

CEulerParticles::~CEulerParticles()
{

}

void CEulerParticles::resize(UInt vNumOfParticles, const Real* vParticlesPos, const Real* vParticlesVel, const Real* vParticlesColor)
{
	m_NumOfParticles = vNumOfParticles;

	if (vParticlesPos == nullptr)
	{
		resizeDeviceVector(m_ParticlesPos, 3 * vNumOfParticles);
		CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_ParticlesPos), 0, 3 * vNumOfParticles * sizeof(Real)));
	}
	else
	{
		assignDeviceVectorReal(m_ParticlesPos, vParticlesPos, vParticlesPos + 3 * vNumOfParticles);
	}

	if (vParticlesVel == nullptr)
	{
		resizeDeviceVector(m_ParticlesVel, 3 * vNumOfParticles);
		CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_ParticlesVel), 0, 3 * vNumOfParticles * sizeof(Real)));
	}
	else
	{
		assignDeviceVectorReal(m_ParticlesVel, vParticlesVel, vParticlesVel + 3 * vNumOfParticles);
	}

	if (vParticlesColor == nullptr)
	{
		resizeDeviceVector(m_ParticlesColor, 3 * vNumOfParticles);
		CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_ParticlesColor), 0, 3 * vNumOfParticles * sizeof(Real)));
	}
	else
	{
		assignDeviceVectorReal(m_ParticlesColor, vParticlesColor, vParticlesColor + 3 * vNumOfParticles);
	}

	resizeDeviceVector(m_ParticlesScalarValue, vNumOfParticles);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_ParticlesScalarValue), 0, vNumOfParticles * sizeof(Real)));
	resizeDeviceVector(m_ParticlesVectorValue, 3 * vNumOfParticles);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_ParticlesVectorValue), 0, 3 * vNumOfParticles * sizeof(Real)));
	resizeDeviceVector(m_ParticlesMidPos, 3 * vNumOfParticles);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_ParticlesMidPos), 0, 3 * vNumOfParticles * sizeof(Real)));
	resizeDeviceVector(m_ParticlesThreeFourthsPos, 3 * vNumOfParticles);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_ParticlesThreeFourthsPos), 0, 3 * vNumOfParticles * sizeof(Real)));
	resizeDeviceVector(m_VelFieldCurPosVel, 3 * vNumOfParticles);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_VelFieldCurPosVel), 0, 3 * vNumOfParticles * sizeof(Real)));
	resizeDeviceVector(m_VelFieldMidPosVel, 3 * vNumOfParticles);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_VelFieldMidPosVel), 0, 3 * vNumOfParticles * sizeof(Real)));
	resizeDeviceVector(m_VelFieldThreeFourthsPosVel, 3 * vNumOfParticles);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_VelFieldThreeFourthsPosVel), 0, 3 * vNumOfParticles * sizeof(Real)));
}

UInt CEulerParticles::getNumOfParticles() const
{
	return m_NumOfParticles;
}

const thrust::device_vector<Real>& CEulerParticles::getParticlesPos() const
{
	return m_ParticlesPos;
}

const thrust::device_vector<Real>& CEulerParticles::getParticlesVel() const
{
	return m_ParticlesVel;
}

thrust::device_vector<Real>& CEulerParticles::getParticlesVel()
{
	return m_ParticlesVel;
}

const thrust::device_vector<Real>& CEulerParticles::getParticlesColor() const
{
	return m_ParticlesColor;
}

const Real* CEulerParticles::getConstParticlesPosGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ParticlesPos);
}

const Real* CEulerParticles::getConstParticlesVelGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ParticlesVel);
}

const Real* CEulerParticles::getConstParticlesColorGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ParticlesColor);
}

Real* CEulerParticles::getParticlesPosGPUPtr()
{
	return getRawDevicePointerReal(m_ParticlesPos);
}

Real* CEulerParticles::getParticlesVelGPUPtr()
{
	return getRawDevicePointerReal(m_ParticlesVel);
}

Real* CEulerParticles::getParticlesColorGPUPtr()
{
	return getRawDevicePointerReal(m_ParticlesColor);
}

void CEulerParticles::setParticlesPos(const Real* vParticlesPosCPUPtr)
{
	assignDeviceVectorReal(m_ParticlesPos, vParticlesPosCPUPtr, vParticlesPosCPUPtr + 3 * m_NumOfParticles);
}

void CEulerParticles::setParticlesVel(const Real* vParticlesVelCPUPtr)
{
	assignDeviceVectorReal(m_ParticlesVel, vParticlesVelCPUPtr, vParticlesVelCPUPtr + 3 * m_NumOfParticles);
}

void CEulerParticles::setParticlesColor(const Real* vParticlesColorCPUPtr)
{
	assignDeviceVectorReal(m_ParticlesColor, vParticlesColorCPUPtr, vParticlesColorCPUPtr + 3 * m_NumOfParticles);
}

void CEulerParticles::generateParticlesInFluid(const CCellCenteredScalarField& vFluidSDF, const CCellCenteredScalarField& vSolidSDF, UInt vNumOfPerGrid)
{
	UInt NumOfFluidGrid = 0;
	Vector3i Resolution = vFluidSDF.getResolution();
	Vector3 Origin = vFluidSDF.getOrigin();
	Vector3 Spacing = vFluidSDF.getSpacing();
	thrust::device_vector<bool> FluidGridFlag;
	resizeDeviceVector(FluidGridFlag, Resolution.x * Resolution.y * Resolution.z);
	
	findFluidGridInvoker(vFluidSDF, vSolidSDF, NumOfFluidGrid, FluidGridFlag);

	thrust::host_vector<bool> FluidGridFlagResult = FluidGridFlag;
	vector<Real> RandPosCPU(3 * vNumOfPerGrid * NumOfFluidGrid, 0);

	srand((Int)time(0));

	UInt FluidGridIndex = 0;
	for (int z = 0; z < Resolution.z; z++)
	{
		for (int y = 0; y < Resolution.y; y++)
		{
			for (int x = 0; x < Resolution.x; x++)
			{
				if (FluidGridFlagResult[z * Resolution.x * Resolution.y + y * Resolution.x + x] == true)
				{
					for (UInt i = 0; i < vNumOfPerGrid; i++)
					{
						UInt CurParticleIndex = FluidGridIndex * vNumOfPerGrid + i;
						RandPosCPU[3 * CurParticleIndex] = (x + ((Real)rand() / (Real)RAND_MAX)) * Spacing.x + Origin.x;
						RandPosCPU[3 * CurParticleIndex + 1] = (y + ((Real)rand() / (Real)RAND_MAX)) * Spacing.y + Origin.y;
						RandPosCPU[3 * CurParticleIndex + 2] = (z + ((Real)rand() / (Real)RAND_MAX)) * Spacing.z + Origin.z;
					}
					FluidGridIndex++;
				}
			}
		}
	}

	resize(NumOfFluidGrid * vNumOfPerGrid, RandPosCPU.data());
}

void CEulerParticles::statisticalFluidDensity(CCellCenteredScalarField& voFluidDensityField)
{
	statisticalFluidDensityInvoker(m_ParticlesPos, voFluidDensityField);
}

void CEulerParticles::transferParticlesScalarValue2Field(CCellCenteredScalarField& voScalarField, CCellCenteredScalarField& voWeightField, EPGTransferAlgorithm vTransferAlg)
{
	tranferParticles2CCSFieldInvoker(m_ParticlesPos, m_ParticlesScalarValue, voScalarField, voWeightField, vTransferAlg);
}

void CEulerParticles::transferParticlesVectorValue2Field(CCellCenteredVectorField& voVectorField, CCellCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg)
{
	tranferParticles2CCVFieldInvoker(m_ParticlesPos, m_ParticlesVectorValue, voVectorField, voWeightField, vTransferAlg);
}

void CEulerParticles::transferParticlesVectorValue2Field(CFaceCenteredVectorField& voVectorField, CFaceCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg)
{
	tranferParticles2FCVFieldInvoker(m_ParticlesPos, m_ParticlesVectorValue, voVectorField, voWeightField, vTransferAlg);
}

void CEulerParticles::transferScalarField2Particles(const CCellCenteredScalarField& vScalarField, EPGTransferAlgorithm vTransferAlg)
{
	tranferCCSField2ParticlesInvoker(m_ParticlesPos, m_ParticlesScalarValue, vScalarField, vTransferAlg);
}

void CEulerParticles::transferVectorField2Particles(const CCellCenteredVectorField& vVectorField, EPGTransferAlgorithm vTransferAlg)
{
	tranferCCVField2ParticlesInvoker(m_ParticlesPos, m_ParticlesVectorValue, vVectorField, vTransferAlg);
}

void CEulerParticles::transferVectorField2Particles(const CFaceCenteredVectorField& vVectorField, EPGTransferAlgorithm vTransferAlg)
{
	tranferFCVField2ParticlesInvoker(m_ParticlesPos, m_ParticlesVectorValue, vVectorField, vTransferAlg);
}

void CEulerParticles::transferParticlesColor2Field(CCellCenteredVectorField& voColorField, CCellCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg)
{
	tranferParticles2CCVFieldInvoker(m_ParticlesPos, m_ParticlesColor, voColorField, voWeightField, vTransferAlg);
}

void CEulerParticles::transferParticlesVel2Field(CFaceCenteredVectorField& voVelField, CFaceCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg)
{
	tranferParticles2FCVFieldInvoker(m_ParticlesPos, m_ParticlesVel, voVelField, voWeightField, vTransferAlg);
}

void CEulerParticles::transferVelField2Particles(const CFaceCenteredVectorField& vVelField, EPGTransferAlgorithm vTransferAlg)
{
	tranferFCVField2ParticlesInvoker(m_ParticlesPos, m_ParticlesVel, vVelField, vTransferAlg);
}

void CEulerParticles::advectParticlesInVelField
(
	const CFaceCenteredVectorField& vVelField, 
	Real vDeltaT, 
	Real vCFLNumber,
	const CCellCenteredScalarField& vBoundarysSDF,
	EAdvectionAccuracy vAdvectionAccuracy,
	ESamplingAlgorithm vSamplingAlg
)
{
	Real SubStepTime = 0.0;
	bool IsFinishedAdvection = false;
	Real MinGridSpacing = min(min(vVelField.getSpacing().x, vVelField.getSpacing().y), vVelField.getSpacing().z);

	while (!IsFinishedAdvection)
	{
		Real DeltaSubStepTime = 0.0;

		vVelField.sampleField(m_ParticlesPos, m_VelFieldCurPosVel, vSamplingAlg);

		Real AbsMaxVelComponent = max(abs(getMaxValue(m_VelFieldCurPosVel)), abs(getMinValue(m_VelFieldCurPosVel)));
		//std::cout << "AbsMaxVelComponent = " << AbsMaxVelComponent << endl;
		if (AbsMaxVelComponent != 0.0)
		{
			DeltaSubStepTime = vCFLNumber * MinGridSpacing / AbsMaxVelComponent;
		}
		else
		{
			DeltaSubStepTime = vDeltaT;
			std::cout << "AbsMaxVelComponent == 0.0 !" << endl;
		}

		if (SubStepTime + DeltaSubStepTime >= vDeltaT)
		{
			DeltaSubStepTime = vDeltaT - SubStepTime;
			IsFinishedAdvection = true;
		}
		else if (SubStepTime + 2 * DeltaSubStepTime >= vDeltaT)
		{
			DeltaSubStepTime = 0.5 * (vDeltaT - SubStepTime);
		}
		else
		{

		}

		if (vAdvectionAccuracy == EAdvectionAccuracy::RK1)
		{
			axpyReal(m_VelFieldCurPosVel, m_ParticlesPos, DeltaSubStepTime);
		}
		else if (vAdvectionAccuracy == EAdvectionAccuracy::RK2)
		{
			Real HalfDeltaSubStepTime = 0.5 * DeltaSubStepTime;

			assignDeviceVectorReal(m_ParticlesPos, m_ParticlesMidPos);

			axpyReal(m_VelFieldCurPosVel, m_ParticlesMidPos, HalfDeltaSubStepTime);
			vVelField.sampleField(m_ParticlesMidPos, m_VelFieldMidPosVel, vSamplingAlg);

			axpyReal(m_VelFieldMidPosVel, m_ParticlesPos, DeltaSubStepTime);
		}
		else if (vAdvectionAccuracy == EAdvectionAccuracy::RK3)
		{
			Real HalfDeltaSubStepTime = 0.5 * DeltaSubStepTime;
			Real ThreeFourthsDeltaSubStepTime = 0.75 * DeltaSubStepTime;
			Real TwoNinthsDeltaSubStepTime = 2.0 / 9.0 * DeltaSubStepTime;
			Real ThreeNinthsDeltaSubStepTime = 3.0 / 9.0 * DeltaSubStepTime;
			Real FourNinthsDeltaSubStepTime = 4.0 / 9.0 * DeltaSubStepTime;

			assignDeviceVectorReal(m_ParticlesPos, m_ParticlesMidPos);
			assignDeviceVectorReal(m_ParticlesPos, m_ParticlesThreeFourthsPos);

			axpyReal(m_VelFieldCurPosVel, m_ParticlesMidPos, HalfDeltaSubStepTime);
			vVelField.sampleField(m_ParticlesMidPos, m_VelFieldMidPosVel, vSamplingAlg);

			axpyReal(m_VelFieldMidPosVel, m_ParticlesThreeFourthsPos, ThreeFourthsDeltaSubStepTime);
			vVelField.sampleField(m_ParticlesThreeFourthsPos, m_VelFieldThreeFourthsPosVel, vSamplingAlg);

			axpyReal(m_VelFieldCurPosVel, m_ParticlesPos, TwoNinthsDeltaSubStepTime);
			axpyReal(m_VelFieldMidPosVel, m_ParticlesPos, ThreeNinthsDeltaSubStepTime);
			axpyReal(m_VelFieldThreeFourthsPosVel, m_ParticlesPos, FourNinthsDeltaSubStepTime);			
		}
		else
		{

		}

		SubStepTime += DeltaSubStepTime;
	}

	if (vBoundarysSDF.getResolution() == vVelField.getResolution())
	{
		__fixParticlesPosWithBoundarys(vBoundarysSDF);
	}
}

void CEulerParticles::__fixParticlesPosWithBoundarys(const CCellCenteredScalarField& vBoundarysSDF)
{
	fixParticlesPosWithBoundarysInvoker(m_ParticlesPos, vBoundarysSDF);
}