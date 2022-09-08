#include "CuDenseVector.h"
#include "CudaContextManager.h"
#include "ThrustWapper.cuh"

CCuDenseVector::CCuDenseVector(UInt vSize)
{
	resizeDeviceVector(m_VectorValue, vSize);

	CHECK_CUDA(cudaMemset(
		getRawDevicePointerReal(m_VectorValue),
		0, 
		vSize * sizeof(Real)
	));
	CUDA_SYN
	m_Size = vSize;
	__freshDenseVectorDescr();
}

CCuDenseVector::CCuDenseVector(const Real* vCPUPtr, UInt vSize)
{
	resizeDeviceVector(m_VectorValue, vSize);

	CHECK_CUDA(cudaMemcpy(
		getRawDevicePointerReal(m_VectorValue),
		vCPUPtr, 
		vSize * sizeof(Real), 
		cudaMemcpyHostToDevice
	));
	CUDA_SYN
	m_Size = vSize;
	__freshDenseVectorDescr();
}

CCuDenseVector::CCuDenseVector(const CCuDenseVector & vOther)
{
	resizeDeviceVector(m_VectorValue, vOther.m_Size);
	CHECK_CUBLAS(cublasCopy(
		CCudaContextManager::getInstance().getCublasHandle(), 
		vOther.m_Size, 
		getReadOnlyRawDevicePointer(vOther.getConstVectorValue()), 
		1,
		getRawDevicePointerReal(m_VectorValue), 
		1
	));
	CUDA_SYN

	m_Size = vOther.getSize();
	__freshDenseVectorDescr();
	
}

CCuDenseVector::~CCuDenseVector()
{
	CHECK_CUSPARSE(cusparseDestroyDnVec(m_CuDenseVectorDescr));
}

void CCuDenseVector::setZero()
{
	CHECK_CUDA(cudaMemset(
		getRawDevicePointerReal(m_VectorValue),
		0,
		m_Size * sizeof(Real)
	));
	CUDA_SYN
}

void CCuDenseVector::updateSize(UInt vSize)
{
	m_Size = vSize;
	__freshDenseVectorDescr();
}

cusparseDnVecDescr_t CCuDenseVector::getCuDenseVectorDescr() const
{
	return m_CuDenseVectorDescr;
}

const thrust::device_vector<Real>& CCuDenseVector::getConstVectorValue() const
{
	return m_VectorValue;
}

thrust::device_vector<Real>& CCuDenseVector::getVectorValue()
{
	return m_VectorValue;
}

const Real* CCuDenseVector::getConstVectorValueGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_VectorValue);
}

Real* CCuDenseVector::getVectorValueGPUPtr()
{
	return getRawDevicePointerReal(m_VectorValue);
}

UInt CCuDenseVector::getSize() const
{
	return m_Size;
}

void CCuDenseVector::__freshDenseVectorDescr()
{
	if (m_CuDenseVectorDescr != nullptr)
	{
		CHECK_CUSPARSE(cusparseDestroyDnVec(m_CuDenseVectorDescr));
		m_CuDenseVectorDescr = nullptr;
	}

	CHECK_CUSPARSE(cusparseCreateDnVec(
		&m_CuDenseVectorDescr,
		m_Size,
		getRawDevicePointerReal(m_VectorValue),
		CUDA_REAL_TYPE
	));
	CUDA_SYN
}

void CCuDenseVector::operator=(const CCuDenseVector & vOther)
{
	_ASSERT(m_Size == vOther.m_Size);

	CHECK_CUBLAS(cublasCopy(
		CCudaContextManager::getInstance().getCublasHandle(),
		vOther.m_Size,
		getReadOnlyRawDevicePointer(vOther.getConstVectorValue()),
		1,
		getRawDevicePointerReal(m_VectorValue),
		1
	));
	CUDA_SYN
}

void CCuDenseVector::operator+=(const CCuDenseVector & vOther)
{
	this->plusAlphaX(vOther, 1);
}

void CCuDenseVector::operator*=(Real vCoefficient)
{
	this->scale(vCoefficient);
}

Real CCuDenseVector::operator*(const CCuDenseVector & vOther)
{
	return this->dot(vOther);
}

Real CCuDenseVector::operator[](int i) const
{
	return getElementReal(m_VectorValue, i);
}

void CCuDenseVector::resize(UInt vSize, Real vFillData)
{
	m_Size = vSize;
	resizeDeviceVector(m_VectorValue, vSize, vFillData);
	__freshDenseVectorDescr();
}

void CCuDenseVector::append(const vector<Real>& vCPUData)
{
	appendSTDVectorToDeviceVectorReal(vCPUData, m_VectorValue);
	m_Size += vCPUData.size();
	__freshDenseVectorDescr();
}

void CCuDenseVector::append(const thrust::device_vector<Real>& vGPUData)
{
	appendDeviceVectorToDeviceVectorReal(vGPUData, m_VectorValue);
	m_Size += vGPUData.size();
	__freshDenseVectorDescr();
}

void CCuDenseVector::plusAlphaX(const CCuDenseVector & vOther, Real vAlpha)
{
	if (abs(vAlpha) < EPSILON) return;

	CHECK_CUBLAS(cublasAxpy(
		CCudaContextManager::getInstance().getCublasHandle(), 
		m_Size, 
		&vAlpha,
		getReadOnlyRawDevicePointer(vOther.getConstVectorValue()),
		1,
		getRawDevicePointerReal(m_VectorValue),
		1
	));
	CUDA_SYN
	
}

void CCuDenseVector::scale(Real vCoefficient)
{
	CHECK_CUBLAS(cublasScal(
		CCudaContextManager::getInstance().getCublasHandle(), 
		m_Size, 
		&vCoefficient, 
		getRawDevicePointerReal(m_VectorValue),
		1
	));
	CUDA_SYN
	
}

Real CCuDenseVector::norm2()
{
	Real Result = 0;

	CHECK_CUBLAS(cublasNrm2(
		CCudaContextManager::getInstance().getCublasHandle(), 
		m_Size, 
		getReadOnlyRawDevicePointer(m_VectorValue),
		1, 
		&Result
	));
	CUDA_SYN
	
	return Result;
}

Real CCuDenseVector::dot(const CCuDenseVector & vOther)
{
	_ASSERT(m_Size == vOther.m_Size);

	Real Result = 0;

	CHECK_CUBLAS(cublasDot(
		CCudaContextManager::getInstance().getCublasHandle(), 
		m_Size, 
		getReadOnlyRawDevicePointer(m_VectorValue),
		1, 
		getReadOnlyRawDevicePointer(vOther.getConstVectorValue()),
		1, 
		&Result
	));
	CUDA_SYN
	
	return Result;
}

void CCuDenseVector::fillValue(const Real & vOther)
{
	fillDeviceVectorReal(m_VectorValue, vOther);
}

ostream & operator<<(ostream & voOut, const CCuDenseVector& vInput)
{
	for (int i = 0; i < vInput.getSize() / 3; i++)
	{
		voOut << "(" << vInput[i * 3] << "," << vInput[i * 3 + 1] << "," << vInput[i * 3 + 2] << ")" << endl;
	}
	return voOut;
}
