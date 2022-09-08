#include "FaceCenteredVectorGrid.h"
#include "CudaContextManager.h"

CFaceCenteredVectorGrid::CFaceCenteredVectorGrid()
{

}

CFaceCenteredVectorGrid::CFaceCenteredVectorGrid
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
		vResolutionX, vResolutionY, vResolutionZ,
		vOriginX, vOriginY, vOriginZ,
		vSpacingX, vSpacingY, vSpacingZ,
		vDataXCPUPtr,
		vDataYCPUPtr,
		vDataZCPUPtr
	);
}

CFaceCenteredVectorGrid::CFaceCenteredVectorGrid
(
	const Vector3i& vResolution,
	const Vector3& vOrigin,
	const Vector3& vSpacing,
	Real* vDataXCPUPtr,
	Real* vDataYCPUPtr,
	Real* vDataZCPUPtr
)
{
	resize(vResolution, vOrigin, vSpacing, vDataXCPUPtr, vDataYCPUPtr, vDataZCPUPtr);
}

CFaceCenteredVectorGrid::CFaceCenteredVectorGrid(const CFaceCenteredVectorGrid& vOther)
{
	resize(vOther);
}

CFaceCenteredVectorGrid::~CFaceCenteredVectorGrid()
{

}

void CFaceCenteredVectorGrid::resize
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

void CFaceCenteredVectorGrid::resize
(
	const Vector3i& vResolution,
	const Vector3& vOrigin,
	const Vector3& vSpacing,
	Real* vDataXCPUPtr,
	Real* vDataYCPUPtr,
	Real* vDataZCPUPtr
)
{
	setGridParameters(vResolution, vOrigin, vSpacing);

	Vector3i AugResX = Vector3i(vResolution.x + 1, vResolution.y, vResolution.z);
	Vector3i AugResY = Vector3i(vResolution.x, vResolution.y + 1, vResolution.z);
	Vector3i AugResZ = Vector3i(vResolution.x, vResolution.y, vResolution.z + 1);

	if (vDataXCPUPtr == nullptr)
	{
		resizeDataX(AugResX);
	}
	else
	{
		resizeDataX(vDataXCPUPtr, vDataXCPUPtr + AugResX.x * AugResX.y * AugResX.z);
	}
	if (vDataYCPUPtr == nullptr)
	{
		resizeDataY(AugResY);
	}
	else
	{
		resizeDataY(vDataYCPUPtr, vDataYCPUPtr + AugResY.x * AugResY.y * AugResY.z);
	}
	if (vDataZCPUPtr == nullptr)
	{
		resizeDataZ(AugResZ);
	}
	else
	{
		resizeDataZ(vDataZCPUPtr, vDataZCPUPtr + AugResZ.x * AugResZ.y * AugResZ.z);
	}
}

void CFaceCenteredVectorGrid::resize(const CFaceCenteredVectorGrid& vOther)
{
	setGridParameters(vOther.getResolution(), vOther.getOrigin(), vOther.getSpacing());

	resizeDataX(vOther.getConstGridDataX());
	resizeDataY(vOther.getConstGridDataY());
	resizeDataZ(vOther.getConstGridDataZ());
}

void CFaceCenteredVectorGrid::scale(Real vScalarValue)
{
	Int AugResSizeX = (getResolution().x + 1) * getResolution().y * getResolution().z;
	Int AugResSizeY = getResolution().x * (getResolution().y + 1) * getResolution().z;
	Int AugResSizeZ = getResolution().x * getResolution().y * (getResolution().z + 1);

	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeX, &vScalarValue, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeY, &vScalarValue, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeZ, &vScalarValue, getGridDataZGPUPtr(), 1));	
}

void CFaceCenteredVectorGrid::scale(Vector3 vVectorValue)
{
	Int AugResSizeX = (getResolution().x + 1) * getResolution().y * getResolution().z;
	Int AugResSizeY = getResolution().x * (getResolution().y + 1) * getResolution().z;
	Int AugResSizeZ = getResolution().x * getResolution().y * (getResolution().z + 1);

	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeX, &vVectorValue.x, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeY, &vVectorValue.y, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasScal(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeZ, &vVectorValue.z, getGridDataZGPUPtr(), 1));
}

void CFaceCenteredVectorGrid::plusAlphaX(const CFaceCenteredVectorGrid& vVectorGrid, Real vScalarValue)
{
	_ASSERTE(getResolution() == vVectorGrid.getResolution());

	if (abs(vScalarValue) < EPSILON) return;

	Int AugResSizeX = (getResolution().x + 1) * getResolution().y * getResolution().z;
	Int AugResSizeY = getResolution().x * (getResolution().y + 1) * getResolution().z;
	Int AugResSizeZ = getResolution().x * getResolution().y * (getResolution().z + 1);

	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeX, &vScalarValue, vVectorGrid.getConstGridDataXGPUPtr(), 1, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeY, &vScalarValue, vVectorGrid.getConstGridDataYGPUPtr(), 1, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeZ, &vScalarValue, vVectorGrid.getConstGridDataZGPUPtr(), 1, getGridDataZGPUPtr(), 1));
}

void CFaceCenteredVectorGrid::plusAlphaX(const CFaceCenteredVectorGrid& vVectorGrid, Vector3 vVectorValue)
{
	_ASSERTE(getResolution() == vVectorGrid.getResolution());

	if (abs(vVectorValue.x) < EPSILON && abs(vVectorValue.y) < EPSILON && abs(vVectorValue.z) < EPSILON) return;

	Int AugResSizeX = (getResolution().x + 1) * getResolution().y * getResolution().z;
	Int AugResSizeY = getResolution().x * (getResolution().y + 1) * getResolution().z;
	Int AugResSizeZ = getResolution().x * getResolution().y * (getResolution().z + 1);

	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeX, &vVectorValue.x, vVectorGrid.getConstGridDataXGPUPtr(), 1, getGridDataXGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeY, &vVectorValue.y, vVectorGrid.getConstGridDataYGPUPtr(), 1, getGridDataYGPUPtr(), 1));
	CHECK_CUBLAS(cublasAxpy(CCudaContextManager::getInstance().getCublasHandle(), AugResSizeZ, &vVectorValue.z, vVectorGrid.getConstGridDataZGPUPtr(), 1, getGridDataZGPUPtr(), 1));
}

void CFaceCenteredVectorGrid::operator=(const CFaceCenteredVectorGrid& vOther)
{
	resize(vOther);
}

void CFaceCenteredVectorGrid::operator+=(const CFaceCenteredVectorGrid& vVectorGrid)
{
	_ASSERTE(getResolution() == vVectorGrid.getResolution());

	this->plusAlphaX(vVectorGrid, 1);
}

void CFaceCenteredVectorGrid::operator*=(Real vScalarValue)
{
	this->scale(vScalarValue);
}
void CFaceCenteredVectorGrid::operator*=(Vector3 vVectorValue)
{
	this->scale(vVectorValue);
}