#pragma once
#include "Common.h"
#include "CubicLagrangeDiscreteGrid.h"

void fillCellRelativeNodeIndexInvoker
(
	UInt vTotalCellNum,
	Vector3ui vRes,
	UInt VertexNum,
	UInt vXEdgeNum,
	UInt vYEdgeNum,
	UInt vZEdgeNum,

	UInt* voCellNodeIndex
);

void interpolateInvoker
(
	const Real* vX,
	UInt vXCount,

	const Real* vNode,
	const UInt* vCell,
	SCLDGridInfo vCLDGridInfo,

	Real* voInterpolateResult,
	Real* voInterpolateGrad,
	Vector3 vGridTransform,
	SMatrix3x3 vGridRotation
);