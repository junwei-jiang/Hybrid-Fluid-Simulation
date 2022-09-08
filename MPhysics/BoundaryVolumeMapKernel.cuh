#pragma once
#include "BoundaryVolumeMap.h"

void sendSDFDataToCLDGridInvoker
(
	SCLDGridInfo vCLDGridInfo,
	Real* voGridNodeValue,
	const UInt* vCellData,
	cudaTextureObject_t vSDFTexture,
	bool vIsInvOutside,
	Vector3 vSDFMin,
	Vector3 vSDFInvCellSize
);

void generateVolumeDataInvoker
(
	SCLDGridInfo vCLDGridInfo,
	Real* voGridNodeValue,
	const UInt* vCellData,
	cudaTextureObject_t vSDFTexture,
	const Real* vSDFField,
	Real vSupportRadius
);

void queryVolumeAndClosestPointInvoker
(
	const Real* vDistData,
	Real* vioQueryPos,
	Real* vioParticleVel,
	UInt vQueryDataSize,
	Real vSupportRadius,
	Real vParticleRadius,

	Real* vioVolumeData,
	Real* vioClosestPos,

	Vector3 vGridTransform,
	SMatrix3x3 vGridRotation,
	SMatrix3x3 vInvGridRotation
);