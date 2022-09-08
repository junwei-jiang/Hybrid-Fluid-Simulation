#pragma once
#include "VectorField.h"
#include "CellCenteredVectorGrid.h"

class CCellCenteredVectorField final : public CCellCenteredVectorGrid, public CVectorField
{
public:
	CCellCenteredVectorField();
	CCellCenteredVectorField
	(
		UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
		Real vOriginX = 0, Real vOriginY = 0, Real vOriginZ = 0,
		Real vSpacingX = 1, Real vSpacingY = 1, Real vSpacingZ = 1,
		Real* vDataXCPUPtr = nullptr,
		Real* vDataYCPUPtr = nullptr,
		Real* vDataZCPUPtr = nullptr
	);
	CCellCenteredVectorField
	(
		Vector3i vResolution,
		Vector3 vOrigin = Vector3(0, 0, 0),
		Vector3 vSpacing = Vector3(1, 1, 1),
		Real* vDataXCPUPtr = nullptr,
		Real* vDataYCPUPtr = nullptr,
		Real* vDataZCPUPtr = nullptr
	);
	CCellCenteredVectorField(const CCellCenteredVectorField& vOther);
	~CCellCenteredVectorField();

	void sampleField(const CCellCenteredVectorField& vSampledAbsPosVectorField, CCellCenteredVectorField& voDstVectorField, ESamplingAlgorithm vSamplingAlg = ESamplingAlgorithm::TRILINEAR) const;
	void sampleField(const thrust::device_vector<Real>& vSampledAbsPos, thrust::device_vector<Real>& voDstData, ESamplingAlgorithm vSamplingAlg = ESamplingAlgorithm::TRILINEAR) const;

	void divergence(CCellCenteredScalarField& voDivergenceField) const override;
	void curl(CCellCenteredVectorField& voCurlField) const override;
};
