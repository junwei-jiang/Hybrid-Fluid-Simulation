#include "ParticleInCell.h"

CParticleInCell::CParticleInCell()
{

}

CParticleInCell::CParticleInCell
(
	const CCellCenteredScalarField& vFluidSDF,
	const CCellCenteredScalarField& vSolidSDF,
	UInt vNumOfPerGrid,
	const Vector3i& vResolution,
	const Vector3& vOrigin,
	const Vector3& vSpacing,
	Real vCFLNumber,
	EPGTransferAlgorithm vPGTransferAlg
)
{
	resizeParticleInCell(vFluidSDF, vSolidSDF, vNumOfPerGrid, vResolution, vOrigin, vSpacing, vCFLNumber, vPGTransferAlg);
}

CParticleInCell::~CParticleInCell()
{

}

void CParticleInCell::resizeParticleInCell
(
	const CCellCenteredScalarField& vFluidSDF,
	const CCellCenteredScalarField& vSolidSDF,
	UInt vNumOfPerGrid,
	const Vector3i& vResolution,
	const Vector3& vOrigin,
	const Vector3& vSpacing,
	Real vCFLNumber,
	EPGTransferAlgorithm vPGTransferAlg
)
{
	m_EulerParticles.generateParticlesInFluid(vFluidSDF, vSolidSDF, vNumOfPerGrid);

	m_CFLNumber = vCFLNumber;
	m_PGTransferAlg = vPGTransferAlg;

	m_CCSWeightField.resize(vResolution, vOrigin, vSpacing);
	m_CCVWeightField.resize(vResolution, vOrigin, vSpacing);
	m_FCVWeightField.resize(vResolution, vOrigin, vSpacing);

	m_ScalarValueFlag = false;
	m_VectorValueFlag = false;

	resizeAdvectionSolver(vResolution);
}

void CParticleInCell::advect
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

void CParticleInCell::advect
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

void CParticleInCell::advect
(
	const CFaceCenteredVectorField& vInputField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CFaceCenteredVectorField& voOutputField,
	EAdvectionAccuracy vEAdvectionAccuracy,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	m_EulerParticles.transferVelField2Particles(vInputField, m_PGTransferAlg);

	m_EulerParticles.advectParticlesInVelField(vVelocityField, vDeltaT, m_CFLNumber, vBoundarysSDF, vEAdvectionAccuracy, m_SamplingAlg);

	m_EulerParticles.transferParticlesVel2Field(voOutputField, m_FCVWeightField, m_PGTransferAlg);
}

const CEulerParticles& CParticleInCell::getEulerParticles() const
{
	return m_EulerParticles;
}