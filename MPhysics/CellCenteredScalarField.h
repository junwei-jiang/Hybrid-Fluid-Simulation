#pragma once
#include "ScalarField.h"
#include "CellCenteredScalarGrid.h"
#include "CellCenteredVectorField.h"

class CCellCenteredScalarField final : public CCellCenteredScalarGrid, public CScalarField
{
public:
	CCellCenteredScalarField();
	CCellCenteredScalarField
	(
		UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
		Real vOriginX = 0, Real vOriginY = 0, Real vOriginZ = 0,
		Real vSpacingX = 1, Real vSpacingY = 1, Real vSpacingZ = 1,
		Real* vDataCPUPtr = nullptr
	);
	CCellCenteredScalarField
	(
		Vector3i vResolution,
		Vector3 vOrigin = Vector3(0, 0, 0),
		Vector3 vSpacing = Vector3(1, 1, 1),
		Real* vDataCPUPtr = nullptr
	);
	CCellCenteredScalarField(const CCellCenteredScalarField& vOther);
	~CCellCenteredScalarField();

	void sampleField(const CCellCenteredVectorField& vSampledAbsPosVectorField, CCellCenteredScalarField& voDstScalarField, ESamplingAlgorithm vSamplingAlg = ESamplingAlgorithm::TRILINEAR) const;
	void sampleField(const thrust::device_vector<Real>& vSampledAbsPos, thrust::device_vector<Real>& voDstData, ESamplingAlgorithm vSamplingAlg = ESamplingAlgorithm::TRILINEAR) const;

	void gradient(CCellCenteredVectorField& voGradientField) const override;
	void laplacian(CCellCenteredScalarField& voLaplacianField) const override;
};