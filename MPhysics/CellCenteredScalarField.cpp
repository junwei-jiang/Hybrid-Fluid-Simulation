#include "CellCenteredScalarField.h"
#include "FieldMathTool.cuh"

CCellCenteredScalarField::CCellCenteredScalarField()
{

}

CCellCenteredScalarField::CCellCenteredScalarField
(
	UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
	Real vOriginX, Real vOriginY, Real vOriginZ,
	Real vSpacingX, Real vSpacingY, Real vSpacingZ,
	Real* vDataCPUPtr
)
{
	resize
	(
		vResolutionX, vResolutionY, vResolutionZ,
		vOriginX, vOriginY, vOriginZ,
		vSpacingX, vSpacingY, vSpacingZ,
		vDataCPUPtr
	);
}

CCellCenteredScalarField::CCellCenteredScalarField
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Real* vDataCPUPtr
)
{
	resize(vResolution, vOrigin, vSpacing, vDataCPUPtr);
}

CCellCenteredScalarField::CCellCenteredScalarField(const CCellCenteredScalarField& vOther)
{
	resize(vOther);
}

CCellCenteredScalarField::~CCellCenteredScalarField()
{

}

void CCellCenteredScalarField::sampleField(const CCellCenteredVectorField& vSampledAbsPosVectorField, CCellCenteredScalarField& voDstScalarField, ESamplingAlgorithm vSamplingAlg) const
{
	sampleCellCenteredScalarFieldInvoker(*this, voDstScalarField, vSampledAbsPosVectorField, vSamplingAlg);
}

void CCellCenteredScalarField::sampleField(const thrust::device_vector<Real>& vSampledAbsPos, thrust::device_vector<Real>& voDstData, ESamplingAlgorithm vSamplingAlg) const
{
	sampleCellCenteredScalarFieldInvoker(*this, voDstData, vSampledAbsPos, vSamplingAlg);
}

void CCellCenteredScalarField::gradient(CCellCenteredVectorField& voGradientField) const
{
	gradientInvoker(*this, voGradientField);
}

void CCellCenteredScalarField::laplacian(CCellCenteredScalarField& voLaplacianField) const
{
	laplacianInvoker(*this, voLaplacianField);
}
