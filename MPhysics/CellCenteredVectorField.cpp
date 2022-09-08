#include "CellCenteredVectorField.h"
#include "FieldMathTool.cuh"

CCellCenteredVectorField::CCellCenteredVectorField()
{

}

CCellCenteredVectorField::CCellCenteredVectorField
(
	UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
	Real vOriginX, Real vOriginY, Real vOriginZ,
	Real vSpacingX, Real vSpacingY, Real vSpacingZ,
	Real* vDataXCPUPtr,
	Real* vDataYCPUPtr,
	Real* vDataZCPUPtr
)
{
	resize
	(
		vResolutionX, vResolutionY, vResolutionZ,
		vOriginX, vOriginY, vOriginZ,
		vSpacingX, vSpacingY, vSpacingZ,
		vDataXCPUPtr,
		vDataYCPUPtr,
		vDataZCPUPtr
	);
}

CCellCenteredVectorField::CCellCenteredVectorField
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Real* vDataXCPUPtr,
	Real* vDataYCPUPtr,
	Real* vDataZCPUPtr
)
{
	resize(vResolution, vOrigin, vSpacing, vDataXCPUPtr, vDataYCPUPtr, vDataZCPUPtr);
}

CCellCenteredVectorField::CCellCenteredVectorField(const CCellCenteredVectorField& vOther)
{
	resize(vOther);
}

CCellCenteredVectorField::~CCellCenteredVectorField()
{

}

void CCellCenteredVectorField::sampleField(const CCellCenteredVectorField& vSampledAbsPosVectorField, CCellCenteredVectorField& voDstVectorField, ESamplingAlgorithm vSamplingAlg) const
{
	sampleCellCenteredVectorFieldInvoker(*this, voDstVectorField, vSampledAbsPosVectorField, vSamplingAlg);
}

void CCellCenteredVectorField::sampleField(const thrust::device_vector<Real>& vSampledAbsPos, thrust::device_vector<Real>& voDstData, ESamplingAlgorithm vSamplingAlg) const
{
	sampleCellCenteredVectorFieldInvoker(*this, voDstData, vSampledAbsPos, vSamplingAlg);
}

void CCellCenteredVectorField::divergence(CCellCenteredScalarField& voDivergenceField) const
{
	divergenceInvoker(*this, voDivergenceField);
}

void CCellCenteredVectorField::curl(CCellCenteredVectorField& voCurlField) const
{
	curlInvoker(*this, voCurlField);
}