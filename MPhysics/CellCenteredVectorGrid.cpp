#include "CellCenteredVectorGrid.h"
#include "CudaContextManager.h"

CCellCenteredVectorGrid::CCellCenteredVectorGrid()
{

}

CCellCenteredVectorGrid::~CCellCenteredVectorGrid()
{

}

void CCellCenteredVectorGrid::resize
(
	UInt vResolutionX, UInt vResolutionY, UInt vResolutionZ,
	Real vOriginX, Real vOriginY, Real vOriginZ,
	Real vSpacingX, Real vSpacingY, Real vSpacingZ,
	Real* vDataXCPUPtr,
	Real* vDataYCPUPtr,
	Real* vDataZCPUPtr
)
{
	resize
	(
		Vector3i(vResolutionX, vResolutionY, vResolutionZ),
		Vector3(vOriginX, vOriginY, vOriginZ),
		Vector3(vSpacingX, vSpacingY, vSpacingZ),
		vDataXCPUPtr,
		vDataYCPUPtr,
		vDataZCPUPtr
	);
}

void CCellCenteredVectorGrid::resize
(
	Vector3i vResolution,
	Vector3 vOrigin,
	Vector3 vSpacing,
	Real* vDataXCPUPtr,
	Real* vDataYCPUPtr,
	Real* vDataZCPUPtr
)
{
	setGridParameters(vResolution, vOrigin, vSpacing);

	if (vDataXCPUPtr == nullptr)
	{
		resizeDataX(vResolution);
	}
	else
	{
		resizeDataX(vDataXCPUPtr, vDataXCPUPtr + vResolution.x * vResolution.y * vResolution.z);
	}
	if (vDataYCPUPtr == nullptr)
	{
		resizeDataY(vResolution);
	}
	else
	{
		resizeDataY(vDataYCPUPtr, vDataYCPUPtr + vResolution.x * vResolution.y * vResolution.z);
	}
	if (vDataZCPUPtr == nullptr)
	{
		resizeDataZ(vResolution);
	}
	else
	{
		resizeDataZ(vDataZCPUPtr, vDataZCPUPtr + vResolution.x * vResolution.y * vResolution.z);
	}
}

void CCellCenteredVectorGrid::resize(const CCellCenteredVectorGrid& vOther)
{
	setGridParameters(vOther.getResolution(), vOther.getOrigin(), vOther.getSpacing());

	resizeDataX(vOther.getConstGridDataX());
	resizeDataY(vOther.getConstGridDataY());
	resizeDataZ(vOther.getConstGridDataZ());
}

void CCellCenteredVectorGrid::scale(Real vScalarValue)
{
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, getGridDataZGPUPtr(), 1));
}

void CCellCenteredVectorGrid::scale(Vector3 vVectorValue)
{
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vVectorValue.x, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vVectorValue.y, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vVectorValue.z, getGridDataZGPUPtr(), 1));
}

void CCellCenteredVectorGrid::plusAlphaX(const CCellCenteredVectorGrid& vVectorGrid, Real vScalarValue)
{
	_ASSERTE(getResolution() == vVectorGrid.getResolution());

	if (abs(vScalarValue) < EPSILON) return;

	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, vVectorGrid.getConstGridDataXGPUPtr(), 1, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, vVectorGrid.getConstGridDataYGPUPtr(), 1, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vScalarValue, vVectorGrid.getConstGridDataZGPUPtr(), 1, getGridDataZGPUPtr(), 1));
}

void CCellCenteredVectorGrid::plusAlphaX(const CCellCenteredVectorGrid& vVectorGrid, Vector3 vVectorValue)
{
	_ASSERTE(getResolution() == vVectorGrid.getResolution());

	if (abs(vVectorValue.x) < EPSILON && abs(vVectorValue.y) < EPSILON && abs(vVectorValue.z) < EPSILON) return;

	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vVectorValue.x, vVectorGrid.getConstGridDataXGPUPtr(), 1, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vVectorValue.y, vVectorGrid.getConstGridDataYGPUPtr(), 1, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), getResolution().x * getResolution().y * getResolution().z, &vVectorValue.z, vVectorGrid.getConstGridDataZGPUPtr(), 1, getGridDataZGPUPtr(), 1));
}

void CCellCenteredVectorGrid::operator=(const CCellCenteredVectorGrid& vOther)
{
	resize(vOther);
}

void CCellCenteredVectorGrid::operator+=(const CCellCenteredVectorGrid& vVectorGrid)
{
	_ASSERTE(getResolution() == vVectorGrid.getResolution());

	this->plusAlphaX(vVectorGrid, 1);
}

void CCellCenteredVectorGrid::operator*=(Real vScalarValue)
{
	this->scale(vScalarValue);
}

void CCellCenteredVectorGrid::operator*=(Vector3 vVectorValue)
{
	this->scale(vVectorValue);
}