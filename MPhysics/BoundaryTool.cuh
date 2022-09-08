#pragma once
#include "EulerMathTool.cuh"

void buildSolidsMarkerInvoker(const vector<CCellCenteredScalarField>& vSolidsSDF, CCellCenteredScalarField& voSolidsMarkerField);

void buildSolidsVelFieldInvoker
(
	const CCellCenteredScalarField& vSolidsMarkerField, 
	const thrust::device_vector<Real>& vSolidsVel,
	CFaceCenteredVectorField& voSolidsVelField
);

void transformFieldInvoker
(
	CCellCenteredVectorField& voDstPostionField,
	const thrust::device_vector<Real>& vTranslation,
	const thrust::device_vector<Real>& vRotation,
	const thrust::device_vector<Real>& vScale,
	Int vCurSolidIndex
);