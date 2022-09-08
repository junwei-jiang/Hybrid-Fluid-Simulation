#pragma once
#include "Common.h"

class CCuDenseVector
{
public:
	CCuDenseVector() = default;
	CCuDenseVector(const CCuDenseVector& vOther);
	CCuDenseVector(UInt vSize);
	CCuDenseVector(const Real* vCPUPtr, UInt vSize);
	~CCuDenseVector();
	void setZero();
	void updateSize(UInt vSize);
	void resize(UInt vSize, Real vFillData = REAL_MAX);
	void append(const vector<Real>& vCPUData);
	void append(const thrust::device_vector<Real>& vGPUData);

	void plusAlphaX(const CCuDenseVector & vOther, Real vAlpha);
	void scale(Real vCoefficient);
	Real norm2();
	Real dot(const CCuDenseVector & vOther);
	void fillValue(const Real& vOther);

	void operator=(const CCuDenseVector& vOther);
	void operator+=(const CCuDenseVector& vOther);
	void operator*=(Real vCoefficient);
	Real operator*(const CCuDenseVector& vOther);

	Real operator[](int i) const;
	cusparseDnVecDescr_t getCuDenseVectorDescr() const;

	const thrust::device_vector<Real>& getConstVectorValue() const;
	thrust::device_vector<Real>& getVectorValue();
	const Real* getConstVectorValueGPUPtr() const;
	Real* getVectorValueGPUPtr();
	UInt getSize() const;

private:
	cusparseDnVecDescr_t m_CuDenseVectorDescr = nullptr;
	thrust::device_vector<Real> m_VectorValue;
	UInt m_Size = 0;

	void __freshDenseVectorDescr();
};

ostream & operator<<(ostream & voOut, const CCuDenseVector& vInput);
