#pragma once
#include "EulerMathTool.cuh"

__device__ Real sampleOneCellInCCSFieldTrilerp
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3 vSrcFieldSpacing,
	Vector3 vSrcFieldOrigin,
	Vector3 vSampledPos
);

__device__ Real getCellCenteredScalarFieldValue
(
	const Real* vSrcScalarFieldDataGPUPtr,
	Vector3i vSrcFieldResolution,
	Vector3i vIndex
);

void sampleCellCenteredScalarFieldInvoker
(
	const CCellCenteredScalarField& vSrcScalarField,
	CCellCenteredScalarField& voDstScalarField,
	const CCellCenteredVectorField& vSampledAbsPosVectorField,
	ESamplingAlgorithm vSamplingAlg
);

void sampleCellCenteredScalarFieldInvoker
(
	const CCellCenteredScalarField& vSrcScalarField,
	thrust::device_vector<Real>& voDstData,
	const thrust::device_vector<Real>& vSampledAbsPos,
	ESamplingAlgorithm vSamplingAlg
);

void sampleCellCenteredVectorFieldInvoker
(
	const CCellCenteredVectorField& vSrcVectorField,
	CCellCenteredVectorField& voDstVectorField,
	const CCellCenteredVectorField& vSampledAbsPosVectorField,
	ESamplingAlgorithm vSamplingAlg
);

void sampleCellCenteredVectorFieldInvoker
(
	const CCellCenteredVectorField& vSrcVectorField,
	thrust::device_vector<Real>& voDstData,
	const thrust::device_vector<Real>& vSampledAbsPos,
	ESamplingAlgorithm vSamplingAlg
);

void sampleFaceCenteredVectorFieldInvoker
(
	const CFaceCenteredVectorField& vSrcVectorField,
	CCellCenteredVectorField& voDstVectorField,
	const CCellCenteredVectorField& vSampledAbsPosVectorField,
	ESamplingAlgorithm vSamplingAlg
);

void sampleFaceCenteredVectorFieldInvoker
(
	const CFaceCenteredVectorField& vSrcVectorField,
	thrust::device_vector<Real>& voDstData,
	const thrust::device_vector<Real>& vSampledAbsPos,
	ESamplingAlgorithm vSamplingAlg
);

void sampleFaceCenteredVectorFieldInvoker
(
	const CFaceCenteredVectorField& vSrcVectorField,
	CFaceCenteredVectorField& voDstVectorField,
	const CCellCenteredVectorField& vSampledAbsPosXVectorField,
	const CCellCenteredVectorField& vSampledAbsPosYVectorField,
	const CCellCenteredVectorField& vSampledAbsPosZVectorField,
	ESamplingAlgorithm vSamplingAlg
);

void gradientInvoker(const CCellCenteredScalarField& vScalarField, CCellCenteredVectorField& voGradientField);

void laplacianInvoker(const CCellCenteredScalarField& vScalarField, CCellCenteredScalarField& voLaplacianField);

void divergenceInvoker(const CCellCenteredVectorField& vVectorField, CCellCenteredScalarField& voDivergenceField);

void divergenceInvoker(const CFaceCenteredVectorField& vVectorField, CCellCenteredScalarField& voDivergenceField);

void curlInvoker(const CCellCenteredVectorField& vVectorField, CCellCenteredVectorField& voCurlField);

void curlInvoker(const CFaceCenteredVectorField& vVectorField, CCellCenteredVectorField& voCurlField);