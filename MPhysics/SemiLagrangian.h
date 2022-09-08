#pragma once
#include "AdvectionSolver.h"

class CSemiLagrangian : public CAdvectionSolver
{
public:
	CSemiLagrangian();
	CSemiLagrangian(const Vector3i& vResolution, const Vector3& vOrigin = Vector3(0, 0, 0), const Vector3& vSpacing = Vector3(1, 1, 1));
	~CSemiLagrangian();
	
	void advect
	(
		const CCellCenteredScalarField& vInputField,
		const CFaceCenteredVectorField& vVelocityField,
		Real vDeltaT,
		CCellCenteredScalarField& voOutputField,
		EAdvectionAccuracy vEAdvectionAccuracy = EAdvectionAccuracy::RK2,
		const CCellCenteredScalarField& vBoundarysSDF = CCellCenteredScalarField()
	)override;

	void advect
	(
		const CCellCenteredVectorField& vInputField,
		const CFaceCenteredVectorField& vVelocityField,
		Real vDeltaT,
		CCellCenteredVectorField& voOutputField,
		EAdvectionAccuracy vEAdvectionAccuracy = EAdvectionAccuracy::RK2,
		const CCellCenteredScalarField& vBoundarysSDF = CCellCenteredScalarField()
	) override;

	void advect
	(
		const CFaceCenteredVectorField& vInputField,
		const CFaceCenteredVectorField& vVelocityField,
		Real vDeltaT,
		CFaceCenteredVectorField& voOutputField,
		EAdvectionAccuracy vEAdvectionAccuracy = EAdvectionAccuracy::RK2,
		const CCellCenteredScalarField& vBoundarysSDF = CCellCenteredScalarField()
	)override;

	void backTrace
	(
		const CCellCenteredVectorField& vInputPosField,
		const CFaceCenteredVectorField& vVelocityField,
		Real vDeltaT,
		CCellCenteredVectorField& voOutputPosField,
		EAdvectionAccuracy vEAdvectionAccuracy = EAdvectionAccuracy::RK2
	);

	void resizeSemiLagrangian(const Vector3i& vResolution, const Vector3& vOrigin = Vector3(0, 0, 0), const Vector3& vSpacing = Vector3(1, 1, 1));

private:
	CCellCenteredVectorField m_AdvectionInputPointPosFieldCC;
	CCellCenteredVectorField m_AdvectionOutputPointPosFieldCC;
	CCellCenteredVectorField m_AdvectionInputPointPosXFieldFC;
	CCellCenteredVectorField m_AdvectionInputPointPosYFieldFC;
	CCellCenteredVectorField m_AdvectionInputPointPosZFieldFC;
	CCellCenteredVectorField m_AdvectionOutputPointPosXFieldFC;
	CCellCenteredVectorField m_AdvectionOutputPointPosYFieldFC;
	CCellCenteredVectorField m_AdvectionOutputPointPosZFieldFC;

	CCellCenteredVectorField m_BackTraceInputPointVelField;
	CCellCenteredVectorField m_BackTraceMidPointPosField;
	CCellCenteredVectorField m_BackTraceMidPointVelField;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointPosField;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointVelField;

	CCellCenteredVectorField m_BackTraceInputPointVelFieldX;
	CCellCenteredVectorField m_BackTraceMidPointPosFieldX;
	CCellCenteredVectorField m_BackTraceMidPointVelFieldX;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointPosFieldX;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointVelFieldX;

	CCellCenteredVectorField m_BackTraceInputPointVelFieldY;
	CCellCenteredVectorField m_BackTraceMidPointPosFieldY;
	CCellCenteredVectorField m_BackTraceMidPointVelFieldY;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointPosFieldY;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointVelFieldY;

	CCellCenteredVectorField m_BackTraceInputPointVelFieldZ;
	CCellCenteredVectorField m_BackTraceMidPointPosFieldZ;
	CCellCenteredVectorField m_BackTraceMidPointVelFieldZ;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointPosFieldZ;
	CCellCenteredVectorField m_BackTraceTwoThirdsPointVelFieldZ;
};

typedef std::shared_ptr<CSemiLagrangian> SemiLagrangianSolverPtr;