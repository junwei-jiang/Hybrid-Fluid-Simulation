#pragma once
#include "CellCenteredScalarField.h"
#include "FaceCenteredVectorField.h"

class CBoundarys
{
public:
	CBoundarys();
	CBoundarys(Vector3i vResolution, Vector3 vOrigin = Vector3(0, 0, 0), Vector3 vSpacing = Vector3(1, 1, 1));

	void resizeBoundarys(Vector3i vResolution, Vector3 vOrigin = Vector3(0, 0, 0), Vector3 vSpacing = Vector3(1, 1, 1));

	void setTotalBoundarysVel(const CFaceCenteredVectorField& vBoundarysVelField);

	int getNumOfBoundarys() const;
	const thrust::device_vector<Real>& getBoundarysVel() const;
	const thrust::device_vector<Real>& getBoundarysTranslation() const;
	const thrust::device_vector<Real>& getBoundarysRotation() const;
	const thrust::device_vector<Real>& getBoundarysScale() const;
	const CCellCenteredScalarField& getOriginalBoundarysSDF(Int vIndex = 0) const;
	const CCellCenteredScalarField& getCurrentBoundarysSDF(Int vIndex = 0) const;
	const CCellCenteredScalarField& getTotalBoundarysMarker() const;
	const CCellCenteredScalarField& getTotalBoundarysSDF() const;
	const CFaceCenteredVectorField& getTotalBoundarysVel() const;

	void addBoundary
	(
		const CCellCenteredScalarField& vNewBoundarySDFField, 
		Vector3 vVelocity = Vector3(0, 0, 0),
		Vector3 vTranslation = Vector3(0, 0, 0), 
		Vector3 vRotation = Vector3(1, 1, 1),
		Vector3 vScale = Vector3(1, 1, 1)
	);

	void updateBoundarys(Real vDeltaT);

private:
	Vector3i m_BoundarysResolution;
	Vector3 m_BoundarysOrigin;
	Vector3 m_BoundarysSpacing;

	Int m_NumOfBoundarys = 0;

	thrust::device_vector<Real> m_BounadrysVel;
	thrust::device_vector<Real> m_BounadrysTranslation;
	thrust::device_vector<Real> m_BounadrysRotation;
	thrust::device_vector<Real> m_BounadrysScale;

	vector<CCellCenteredScalarField> m_OriginalBoundarysSDF;
	vector<CCellCenteredScalarField> m_CurrentBoundarysSDF;

	CCellCenteredScalarField m_TotalBoundarysMarker;
	CCellCenteredScalarField m_TotalBoundarysSDF;
	CFaceCenteredVectorField m_TotalBoundarysVel;

	CCellCenteredVectorField m_SamplingPosField;

	void __moveBoundarys(Real vDeltaT);
	void __resamplingCurrentBoundarysSDF();
	void __updateTotalBoundarysMarker();
	void __updateTotalBoundarysSDF();
	void __updateTotalBoundarysVel();
};