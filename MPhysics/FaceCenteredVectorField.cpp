#include "FaceCenteredVectorField.h"
#include "FieldMathTool.cuh"

CFaceCenteredVectorField::CFaceCenteredVectorField()
{

}

CFaceCenteredVectorField::CFaceCenteredVectorField
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

CFaceCenteredVectorField::CFaceCenteredVectorField
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

CFaceCenteredVectorField::CFaceCenteredVectorField(const CFaceCenteredVectorField& vOther)
{
	resize(vOther);
}

CFaceCenteredVectorField::~CFaceCenteredVectorField()
{

}

void CFaceCenteredVectorField::sampleField
(
	const CCellCenteredVectorField& vSampledAbsPosVectorField,
	CCellCenteredVectorField& voDstVectorField,
	ESamplingAlgorithm vSamplingAlg
) const
{
	sampleFaceCenteredVectorFieldInvoker(*this, voDstVectorField, vSampledAbsPosVectorField, vSamplingAlg);
}

void CFaceCenteredVectorField::sampleField
(
	const thrust::device_vector<Real>& vSampledAbsPos,
	thrust::device_vector<Real>& voDstData,
	ESamplingAlgorithm vSamplingAlg
) const
{
	sampleFaceCenteredVectorFieldInvoker(*this, voDstData, vSampledAbsPos, vSamplingAlg);
}

void CFaceCenteredVectorField::sampleField
(
	const CCellCenteredVectorField& vSampledAbsPosXVectorField,
	const CCellCenteredVectorField& vSampledAbsPosYVectorField,
	const CCellCenteredVectorField& vSampledAbsPosZVectorField,
	CFaceCenteredVectorField& voDstVectorField,
	ESamplingAlgorithm vSamplingAlg
) const
{
	sampleFaceCenteredVectorFieldInvoker(*this, voDstVectorField, vSampledAbsPosXVectorField, vSampledAbsPosYVectorField, vSampledAbsPosZVectorField, vSamplingAlg);
}

void CFaceCenteredVectorField::divergence(CCellCenteredScalarField& voDivergenceField) const
{
	divergenceInvoker(*this, voDivergenceField);
}

void CFaceCenteredVectorField::curl(CCellCenteredVectorField& voCurlField) const
{
	curlInvoker(*this, voCurlField);
}