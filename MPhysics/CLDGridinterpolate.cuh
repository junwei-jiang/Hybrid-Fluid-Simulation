#pragma once
#include "Common.h"
#include "CubicLagrangeDiscreteGrid.h"

__host__ __device__ void genShapeFunction(const Vector3& vXi, Vector3 voDN[32], Real voN[32]);

__host__ __device__ void interpolateSinglePos
(
	const Vector3& vX,
	const Real* vNodeGPUPtr,
	const UInt* vCellGPUPtr,
	SCLDGridInfo vCLDGridInfo,

	Real& voValueInte,
	Vector3* voGradInte = nullptr
);