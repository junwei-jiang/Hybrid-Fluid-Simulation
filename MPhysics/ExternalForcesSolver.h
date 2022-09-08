#pragma once
#include "CellCenteredScalarField.h"
#include "CellCenteredVectorField.h"
#include "FaceCenteredVectorField.h"

class CExternalForcesSolver
{
public:
	CExternalForcesSolver();
	~CExternalForcesSolver();
	CExternalForcesSolver(const Vector3i& vResolution, const Vector3& vOrigin = Vector3(0, 0, 0), const Vector3& vSpacing = Vector3(1, 1, 1));

	void addExternalForces(Vector3 vExternalForces);

	void applyExternalForces
	(
		CFaceCenteredVectorField& vioVelField,
		Real vDeltaT,
		const CCellCenteredScalarField& vBoundarySDF = CCellCenteredScalarField()
	);

	void resizeExternalForcesSolver(const Vector3i& vResolution, const Vector3& vOrigin = Vector3(0, 0, 0), const Vector3& vSpacing = Vector3(1, 1, 1));

private:
	Vector3 m_Gravity = Vector3(0, -9.8, 0);
	Vector3 m_ExternalForces = Vector3(0, 0, 0);

	CFaceCenteredVectorField m_UnitFieldFCV;
};

typedef std::shared_ptr<CExternalForcesSolver> ExternalForcesSolverPtr;
