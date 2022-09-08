#pragma once
#include "EulerMathTool.cuh"

void findFluidGridInvoker
(
	const CCellCenteredScalarField& vFluidSDF, 
	const CCellCenteredScalarField& vSolidSDF, 
	UInt& voNumOfFluidGrid, 
	thrust::device_vector<bool>& voFluidGridFlag
);

void buildFluidMarkersInvoker
(
	const CCellCenteredScalarField& vSolidSDF, 
	const thrust::device_vector<Real>& vParticlesPos,
	CCellCenteredScalarField& voMarkersField
);

void statisticalFluidDensityInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	CCellCenteredScalarField& voFluidDensityField
);

void tranferParticles2CCSFieldInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	const thrust::device_vector<Real>& vParticlesScalarValue,
	CCellCenteredScalarField& voScalarField,
	CCellCenteredScalarField& voWeightField,
	EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR,
	bool vIsNormalization = true
);

void tranferParticles2CCVFieldInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	const thrust::device_vector<Real>& vParticlesVectorValue,
	CCellCenteredVectorField& voVectorField,
	CCellCenteredVectorField& voWeightField,
	EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR,
	bool vIsNormalization = true
);

void tranferParticles2FCVFieldInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	const thrust::device_vector<Real>& vParticlesVectorValue,
	CFaceCenteredVectorField& voVectorField,
	CFaceCenteredVectorField& voWeightField,
	EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR,
	bool vIsNormalization = true
);

void tranferCCSField2ParticlesInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	thrust::device_vector<Real>& voParticlesScalarValue,
	const CCellCenteredScalarField& vScalarField,
	EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR
);

void tranferCCVField2ParticlesInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	thrust::device_vector<Real>& voParticlesVectorValue,
	const CCellCenteredVectorField& vVectorField,
	EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR
);

void tranferFCVField2ParticlesInvoker
(
	const thrust::device_vector<Real>& vParticlesPos,
	thrust::device_vector<Real>& voParticlesVectorValue,
	const CFaceCenteredVectorField& vVectorField,
	EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR
);

void fixParticlesPosWithBoundarysInvoker
(
	thrust::device_vector<Real>& vioParticlesPos,
	const CCellCenteredScalarField& vBoundarysSDF
);