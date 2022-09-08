#pragma once
#include "Common.h"
#include "CellCenteredScalarField.h"
#include "FaceCenteredVectorField.h"
#include "CubicLagrangeDiscreteGrid.h"
#include "EulerMathTool.cuh"
#include "KNNSearch.h"

void mixParitcleVelWithPICAndFLIPInvoker
(
	const Real* vPICVel,
	const Real* vFLIPVel,
	const Real* vDistP,
	UInt vParticleSize,
	Real vCellSpace,
	Real vGRate,
	Real vGFRate,
	Real vFRate,
	Real vPRate,

	Real* voPredictVel
);

void samplingSDFFromCLDGridInvoker
(
	std::shared_ptr<CCubicLagrangeDiscreteGrid> vCLDGrid,
	CCellCenteredScalarField& voSDFField
);

void mixFieldWithDensityInvoker
(
	const CCellCenteredScalarField& vScalarFieldA,
	CCellCenteredScalarField& voScalarFieldB,
	const CCellCenteredScalarField& vWeightFieldA,
	const CCellCenteredScalarField& vWeightFieldB
);

void mixFieldWithDensityInvoker
(
	const CCellCenteredVectorField& vVectorFieldA,
	CCellCenteredVectorField& voVectorFieldB,
	const CCellCenteredScalarField& vWeightFieldA,
	const CCellCenteredScalarField& vWeightFieldB
);

void mixFieldWithDensityInvoker
(
	const CFaceCenteredVectorField& vVectorFieldA,
	CFaceCenteredVectorField& voVectorFieldB,
	const CCellCenteredScalarField& vWeightFieldA,
	const CCellCenteredScalarField& vWeightFieldB
);

void buildFluidOutsideSDFInvoker
(
	const CCellCenteredScalarField& vFluidDensityField,
	const CCellCenteredScalarField& vSolidDomainField,
	CCellCenteredScalarField& voFluidOutsideSDFField,
	UInt vExtrapolationDistance
);

void buildFluidInsideSDFInvoker
(
	const CCellCenteredScalarField& vFluidDensityField,
	const CCellCenteredScalarField& vSolidDomainField,
	CCellCenteredScalarField& voFluidInsideSDFField,
	UInt vExtrapolationDistance
);

void buildMixedFluidOutsideSDFInvoker
(
	const CCellCenteredScalarField& vGridFluidDensityField,
	const CCellCenteredScalarField& vMixedFluidDensityField,
	const CCellCenteredScalarField& vGridFluidOutsideSDFField,
	CCellCenteredScalarField& voMixedFluidOutsideSDFField
);

void genParticleByPoissonDiskInvoker
(
	const Real* vDistGridC,
	const Real* vRhoGridC,
	Real* vioRhoGridG,

	Vector3i vGridResolution,
	Vector3 vGridMin,
	UInt vGridDataSize,
	Real vCellSpace,

	const UInt* vCellOffset,
	const UInt* vCellParticleCounts,
	SGridInfo vHashGridInfo,

	UInt vSampleK,
	Real vCreateDistRate,
	Real vRhoMin,
	Real vFluidRestDensity,

	std::shared_ptr<CParticleGroup> voParticleGroup,
	thrust::device_vector<UInt>& voNeedGenNewParticleCellIndexCache
);

void deleteParticleUnderWater
(
	const Real* vDistPC,
	const Real* vParticlePos,
	Real* vioRhoGridG,

	/*上述网格的属性*/
	Vector3i vGridResolution,
	Vector3 vGridMin,
	UInt vGridDataSize,
	Real vCellSpace,

	UInt vParticleSize,
	Real vCellSize,
	Real vDeleteRate,
	Real vTimeStep,
	Real vMaxLiveTime,

	thrust::device_vector<bool>& voShouldDelete,
	std::shared_ptr<CParticleGroup> voTarget
);

void predictParticleVelInvoker
(
	thrust::device_vector<Real>& voVel,
	UInt vParticleSize,
	Vector3 vExtAcc,
	Real vDeltaTime
);