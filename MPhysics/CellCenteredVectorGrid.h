#pragma once
#include "VectorGrid.h"

class CCellCenteredVectorGrid : public CVectorGrid
{
public:
	CCellCenteredVectorGrid();
	virtual ~CCellCenteredVectorGrid();

	void resize
	(
		UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
		Real vOriginX = 0, Real vOriginY = 0, Real vOriginZ = 0,
		Real vSpacingX = 1, Real vSpacingY = 1, Real vSpacingZ = 1,
		Real* vDataXCPUPtr = nullptr,
		Real* vDataYCPUPtr = nullptr,
		Real* vDataZCPUPtr = nullptr
	);
	void resize
	(
		Vector3i vResolution,
		Vector3 vOrigin = Vector3(0, 0, 0),
		Vector3 vSpacing = Vector3(1, 1, 1),
		Real* vDataXCPUPtr = nullptr,
		Real* vDataYCPUPtr = nullptr,
		Real* vDataZCPUPtr = nullptr
	);
	void resize(const CCellCenteredVectorGrid& vOther);

	void scale(Real vScalarValue);
	void scale(Vector3 vVectorValue);
	void plusAlphaX(const CCellCenteredVectorGrid& vVectorGrid, Real vScalarValue);
	void plusAlphaX(const CCellCenteredVectorGrid& vVectorGrid, Vector3 vVectorValue);

	void operator=(const CCellCenteredVectorGrid& vOther);
	void operator+=(const CCellCenteredVectorGrid& vVectorGrid);
	void operator*=(Real vScalarValue);
	void operator*=(Vector3 vVectorValue);
};