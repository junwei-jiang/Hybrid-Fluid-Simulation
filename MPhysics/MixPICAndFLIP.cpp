#include "MixPICAndFLIP.h"
#include "EulerParticlesTool.cuh"

CMixPICAndFLIP::CMixPICAndFLIP()
{

}

CMixPICAndFLIP::CMixPICAndFLIP
(
	const CCellCenteredScalarField& vFluidSDF,
	const CCellCenteredScalarField& vSolidSDF,
	UInt vNumOfPerGrid,
	const Vector3i& vResolution,
	const Vector3& vOrigin,
	const Vector3& vSpacing,
	Real vCFLNumber,
	Real vMixingCoefficient,
	EPGTransferAlgorithm vPGTransferAlg
)
{
	resizeMixPICAndFLIP(vFluidSDF, vSolidSDF, vNumOfPerGrid, vResolution, vOrigin, vSpacing, vCFLNumber, vMixingCoefficient, vPGTransferAlg);
}

CMixPICAndFLIP::~CMixPICAndFLIP()
{

}

void CMixPICAndFLIP::resizeMixPICAndFLIP
(
	const CCellCenteredScalarField& vFluidSDF,
	const CCellCenteredScalarField& vSolidSDF,
	UInt vNumOfPerGrid,
	const Vector3i& vResolution,
	const Vector3& vOrigin,
	const Vector3& vSpacing,
	Real vCFLNumber,
	Real vMixingCoefficient,
	EPGTransferAlgorithm vPGTransferAlg
)
{
	m_EulerParticles.generateParticlesInFluid(vFluidSDF, vSolidSDF, vNumOfPerGrid);
	m_MixingCoefficient = vMixingCoefficient;

	m_CFLNumber = vCFLNumber;
	m_PGTransferAlg = vPGTransferAlg;

	m_CCSWeightField.resize(vResolution, vOrigin, vSpacing);
	m_CCVWeightField.resize(vResolution, vOrigin, vSpacing);
	m_FCVWeightField.resize(vResolution, vOrigin, vSpacing);

	m_DeltaGridVelField.resize(vResolution, vOrigin, vSpacing);
	resizeDeviceVector(m_DeltaParticlesVel, 3 * m_EulerParticles.getNumOfParticles(), 0.0);
	resizeDeviceVector(m_TempParticlesPICVel, 3 * m_EulerParticles.getNumOfParticles(), 0.0);

	m_ScalarValueFlag = false;
	m_VectorValueFlag = false;
	m_VelFlag = false;

	resizeAdvectionSolver(vResolution);
}

void CMixPICAndFLIP::advect
(
	const CCellCenteredScalarField& vInputField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CCellCenteredScalarField& voOutputField,
	EAdvectionAccuracy vEAdvectionAccuracy,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	if (!m_ScalarValueFlag)
	{
		m_EulerParticles.transferScalarField2Particles(vInputField, m_PGTransferAlg);
		m_ScalarValueFlag = true;
	}

	m_EulerParticles.transferParticlesScalarValue2Field(voOutputField, m_CCSWeightField, m_PGTransferAlg);
}

void CMixPICAndFLIP::advect
(
	const CCellCenteredVectorField& vInputField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CCellCenteredVectorField& voOutputField,
	EAdvectionAccuracy vEAdvectionAccuracy,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	if (!m_VectorValueFlag)
	{
		m_EulerParticles.transferVectorField2Particles(vInputField, m_PGTransferAlg);
		m_VectorValueFlag = true;
	}

	m_EulerParticles.transferParticlesVectorValue2Field(voOutputField, m_CCVWeightField, m_PGTransferAlg);
}

void CMixPICAndFLIP::advect
(
	const CFaceCenteredVectorField& vInputField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CFaceCenteredVectorField& voOutputField,
	EAdvectionAccuracy vEAdvectionAccuracy,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	tranferFCVField2ParticlesInvoker(m_EulerParticles.getParticlesPos(), m_TempParticlesPICVel, vInputField, m_PGTransferAlg);

	if (!m_VelFlag)
	{
		m_DeltaGridVelField = vInputField;
		m_VelFlag = true;
	}
	else
	{
		m_DeltaGridVelField.plusAlphaX(vInputField, -1.0);
		m_DeltaGridVelField *= -1.0;
	}

	tranferFCVField2ParticlesInvoker(m_EulerParticles.getParticlesPos(), m_DeltaParticlesVel, m_DeltaGridVelField, m_PGTransferAlg);
	axpyReal(m_DeltaParticlesVel, m_EulerParticles.getParticlesVel(), 1.0);

	scaleReal(m_EulerParticles.getParticlesVel(), (1.0 - m_MixingCoefficient));
	axpyReal(m_TempParticlesPICVel, m_EulerParticles.getParticlesVel(), m_MixingCoefficient);

	m_EulerParticles.advectParticlesInVelField(vVelocityField, vDeltaT, m_CFLNumber, vBoundarysSDF, vEAdvectionAccuracy, m_SamplingAlg);

	m_EulerParticles.transferParticlesVel2Field(voOutputField, m_FCVWeightField, m_PGTransferAlg);

	m_DeltaGridVelField = voOutputField;
}

const CEulerParticles& CMixPICAndFLIP::getEulerParticles() const
{
	return m_EulerParticles;
}