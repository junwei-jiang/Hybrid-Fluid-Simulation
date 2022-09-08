#pragma once
#include "ScalarGrid.h"

class CCellCenteredScalarGrid : public CScalarGrid
{
public:
	CCellCenteredScalarGrid();
	virtual ~CCellCenteredScalarGrid();

	void resize
	(
		UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
		Real vOriginX = 0, Real vOriginY = 0, Real vOriginZ = 0,
		Real vSpacingX = 1, Real vSpacingY = 1, Real vSpacingZ = 1,
		Real* vDataCPUPtr = nullptr
	);
	void resize
	(
		Vector3i vResolution,
		Vector3 vOrigin = Vector3(0, 0, 0),
		Vector3 vSpacing = Vector3(1, 1, 1),
		Real* vDataCPUPtr = nullptr
	);
	void resize(const CCellCenteredScalarGrid& vOther);

	void scale(Real vScalarValue);
	void plusAlphaX(const CCellCenteredScalarGrid& vScalarGrid, Real vScalarValue);

	void operator=(const CCellCenteredScalarGrid& vOther);
	void operator+=(const CCellCenteredScalarGrid& vScalarGrid);
	void operator*=(Real vScalarValue);
};