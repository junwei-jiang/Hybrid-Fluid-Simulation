#include "CellCenteredScalarGrid.h"
#include "CudaContextManager.h"

CCellCenteredScalarGrid::CCellCenteredScalarGrid()
{

}

CCellCenteredScalarGrid::~CCellCenteredScalarGrid()
{

}

void CCellCenteredScalarGrid::resize
(
	UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
	Real vOriginX, Real vOriginY, Real vOriginZ,
	Real vSpacingX, Real vSpacingY, Real vSpacingZ,
	Real* vDataCPUPtr
)
{
	resize
	(
		Vector3i(vResolutionX, vResolutionY, vResolutionZ),
		Vector3(vOriginX, vOriginY, vOriginZ),
		Vector3(vSpacingX, vSpacingY, vSpacingZ),
		vDataCPUPtr
	);
}

void CCellCenteredScalarGrid::resize
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Real* vDataCPUPtr
)
{
	setGridParameters(vResolution, vOrigin, vSpacing);

	if (vDataCPUPtr == nullptr)
	{
		resizeData(vResolution);
	}
	else
	{
		resizeData(vDataCPUPtr, vDataCPUPtr + vResolution.x * vResolution.y * vResolution.z);
	}
}

void CCellCenteredScalarGrid::resize(const CCellCenteredScalarGrid& vOther)
{
	setGridParameters(vOther.getResolution(), vOther.getOrigin(), vOther.getSpacing());

	resizeData(vOther.getConstGridData());
}

void CCellCenteredScalarGrid::scale(Real vScalarValue)
{
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, getGridDataGPUPtr(), 1));
}

void CCellCenteredScalarGrid::plusAlphaX(const CCellCenteredScalarGrid& vScalarGrid, Real vScalarValue)
{
	_ASSERTE(getResolution() == vScalarGrid.getResolution());

	if (abs(vScalarValue) < EPSILON) return;

	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, vScalarGrid.getConstGridDataGPUPtr(), 1, getGridDataGPUPtr(), 1));
}

void CCellCenteredScalarGrid::operator=(const CCellCenteredScalarGrid& vOther)
{
	resize(vOther);
}

void CCellCenteredScalarGrid::operator+=(const CCellCenteredScalarGrid& vScalarGrid)
{
	_ASSERTE(getResolution() == vScalarGrid.getResolution());

	this->plusAlphaX(vScalarGrid, 1);
}

void CCellCenteredScalarGrid::operator*=(Real vScalarValue)
{
	this->scale(vScalarValue);
}
