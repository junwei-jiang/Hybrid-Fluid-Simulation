#pragma once
#include "EulerMathTool.cuh"

void fixFluidDomainInvoker(const CCellCenteredScalarField& vSolidSDF, CCellCenteredScalarField& vioFluidDomain);

void buildFluidMarkersInvoker(const CCellCenteredScalarField& vSolidSDF, const CCellCenteredScalarField& vFluidSDF, CCellCenteredScalarField& voMarkersField);

void buildPressureFdmMatrixAInvoker(Vector3i vResolution, Vector3 vScale, const CCellCenteredScalarField& vMarkersField, thrust::device_vector<Real>& voFdmMatrix);

void buildPressureVectorbInvoker
(
	const CFaceCenteredVectorField& vFluidVelField,
	const CCellCenteredScalarField& vVelDivergenceField,
	const CCellCenteredScalarField& vMarkersField,
	const CFaceCenteredVectorField& vSolidVelField,
	CCuDenseVector& voVectorb
);

void fdmMatrixVectorMulInvoker(Vector3i vResolution, const thrust::device_vector<Real>& vFdmMatrix, const CCuDenseVector& vInputVector, CCuDenseVector& voOutputVector);

void applyPressureGradientInvoker
(
	Vector3i vResolution,
	Vector3 vScale,
	const CCellCenteredScalarField& vMarkersField,
	const Real* vPressureFieldDataGPUPtr,
	CFaceCenteredVectorField& vioFluidVelField,
	const CFaceCenteredVectorField& vSolidVelField
);

void buildFluidSDFInvoker
(
	const CCellCenteredScalarField& vFluidDomainField,
	CCellCenteredScalarField& voFluidSDFField,
	UInt vExtrapolationDistance
);

void buildFluidDensityInvoker
(
	const CCellCenteredScalarField& vFluidDomainField, 
	const CCellCenteredScalarField& vSolidDomainField,
	CCellCenteredScalarField& voFluidDensityField
);

void extrapolatingDataInvoker(CCellCenteredScalarField& vioScalarField, CCellCenteredScalarField& voDisMarkersField, UInt vExtrapolationDistance);

void extrapolatingDataInvoker(CCellCenteredVectorField& vioVectorField, CCellCenteredVectorField& voDisMarkersField, UInt vExtrapolationDistance);

void extrapolatingDataInvoker(CFaceCenteredVectorField& vioVectorField, CFaceCenteredVectorField& voDisMarkersField, UInt vExtrapolationDistance);

void fillSDFFieldInvoker(cudaTextureObject_t vSDFTexture, CCellCenteredScalarField& voSDFField, bool vIsInvSign);

void generateFluidDomainFromBBoxInvoker(Vector3 vMin, Vector3 vMax, CCellCenteredScalarField& voFluidDomainField);