#pragma once
#include "CellCenteredScalarField.h"
#include "CellCenteredVectorField.h"
#include "FaceCenteredVectorField.h"

class CAdvectionSolver
{
public:
	CAdvectionSolver();
	virtual ~CAdvectionSolver();

	virtual void advect
	(
		const CCellCenteredScalarField& vInputField,
		const CFaceCenteredVectorField& vVelocityField,
		Real vDeltaT,
		CCellCenteredScalarField& voOutputField,
		EAdvectionAccuracy vEAdvectionAccuracy,
		const CCellCenteredScalarField& vBoundarysSDF
	) = 0;

	virtual void advect
	(
		const CCellCenteredVectorField& vInputField,
		const CFaceCenteredVectorField& vVelocityField,
		Real vDeltaT,
		CCellCenteredVectorField& voOutputField,
		EAdvectionAccuracy vEAdvectionAccuracy,
		const CCellCenteredScalarField& vBoundarysSDF
	) = 0;

	virtual void advect
	(
		const CFaceCenteredVectorField& vInputField,
		const CFaceCenteredVectorField& vVelocityField,
		Real vDeltaT,
		CFaceCenteredVectorField& voOutputField,
		EAdvectionAccuracy vEAdvectionAccuracy,
		const CCellCenteredScalarField& vBoundarysSDF
	) = 0;

	void resizeAdvectionSolver(const Vector3i& vResolution, ESamplingAlgorithm vSamplingAlg = ESamplingAlgorithm::TRILINEAR);

	void setSamplingAlg(ESamplingAlgorithm vSamplingAlg);

protected:
	bool m_IsInit = false;

	Vector3i m_Resolution;
	ESamplingAlgorithm m_SamplingAlg;
};

typedef std::shared_ptr<CAdvectionSolver> AdvectionSolverPtr;