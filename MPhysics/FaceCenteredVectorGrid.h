#pragma once
#include "VectorGrid.h"
//#include "CellCenteredVectorGrid.h"

class CFaceCenteredVectorGrid : public CVectorGrid
{
public:
	CFaceCenteredVectorGrid();
	CFaceCenteredVectorGrid
	(
		UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
		Real vOriginX = 0, Real vOriginY = 0, Real vOriginZ = 0,
		Real vSpacingX = 1, Real vSpacingY = 1, Real vSpacingZ = 1,
		Real* vDataXCPUPtr = nullptr,
		Real* vDataYCPUPtr = nullptr,
		Real* vDataZCPUPtr = nullptr
	);
	CFaceCenteredVectorGrid
	(
		const Vector3i& vResolution,
		const Vector3& vOrigin = Vector3(0, 0, 0),
		const Vector3& vSpacing = Vector3(1, 1, 1),
		Real* vDataXCPUPtr = nullptr,
		Real* vDataYCPUPtr = nullptr,
		Real* vDataZCPUPtr = nullptr
	);
	CFaceCenteredVectorGrid(const CFaceCenteredVectorGrid& vOther);
	virtual ~CFaceCenteredVectorGrid();

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
		const Vector3i& vResolution,
		const Vector3& vOrigin = Vector3(0, 0, 0),
		const Vector3& vSpacing = Vector3(1, 1, 1),
		Real* vDataXCPUPtr = nullptr,
		Real* vDataYCPUPtr = nullptr,
		Real* vDataZCPUPtr = nullptr
	);
	void resize(const CFaceCenteredVectorGrid& vOther);

	void scale(Real vScalarValue);
	void scale(Vector3 vVectorValue);
	void plusAlphaX(const CFaceCenteredVectorGrid& vVectorGrid, Real vScalarValue);
	void plusAlphaX(const CFaceCenteredVectorGrid& vVectorGrid, Vector3 vVectorValue);

	void operator=(const CFaceCenteredVectorGrid& vOther);
	void operator+=(const CFaceCenteredVectorGrid& vVectorGrid);
	void operator*=(Real vScalarValue);
	void operator*=(Vector3 vVectorValue);
};