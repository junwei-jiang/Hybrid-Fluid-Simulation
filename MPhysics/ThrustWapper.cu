#include "ThrustWapper.cuh"
#include "CudaContextManager.h"

void clearDeviceVectorReal(thrust::device_vector<Real>& vioSource)
{
	vioSource.clear();
}

void push_backDeviceVectorReal(thrust::device_vector<Real>& vioSource, Real vValue)
{
	vioSource.push_back(vValue);
}

UInt getDeviceVectorSize(const thrust::device_vector<Real>& vSource)
{
	return static_cast<UInt>(vSource.size());
}

UInt getDeviceVectorSize(const thrust::device_vector<UInt>& vSource)
{
	return static_cast<UInt>(vSource.size());
}

UInt getDeviceVectorSize(const thrust::device_vector<Int>& vSource)
{
	return static_cast<UInt>(vSource.size());
}

UInt getDeviceVectorSize(const thrust::device_vector<Vector3>& vSource)
{
	return static_cast<UInt>(vSource.size());
}

void appendSTDVectorToDeviceVectorReal(const vector<Real>& vSTDVector, thrust::device_vector<Real>& voSource)
{
	if (voSource.size() == 0)
	{
		sendSTDVectToDeviceReal(vSTDVector, voSource);
	}
	else
	{
		voSource.reserve(vSTDVector.size());
		voSource.insert(voSource.end(), vSTDVector.begin(), vSTDVector.end());
	}
}

void appendDeviceVectorToDeviceVectorReal(const thrust::device_vector<Real>& vDeviceVector, thrust::device_vector<Real>& voSource)
{
	if (voSource.size() == 0)
	{
		voSource = vDeviceVector;
	}
	{
		voSource.reserve(vDeviceVector.size());
		voSource.insert(voSource.end(), vDeviceVector.begin(), vDeviceVector.end());
	}
}

void assignDeviceVectorReal(thrust::device_vector<Real>& voTarget, const Real * vDataStartPtr, const Real * vDataEndPtr)
{
	voTarget.assign(vDataStartPtr, vDataEndPtr);
}

void assignDeviceVectorInt(thrust::device_vector<Int>& voTarget, const Int * vDataStartPtr, const Int * vDataEndPtr)
{
	voTarget.assign(vDataStartPtr, vDataEndPtr);
}

void assignDeviceVectorReal(const thrust::device_vector<Real>& vSource, thrust::device_vector<Real>& voTarget)
{
	voTarget = vSource;
}

void resizeDeviceVector(thrust::device_vector<UInt>& voTarget, UInt vNewSize, UInt vFillData)
{
	voTarget.resize(vNewSize, vFillData);
}

void resizeDeviceVector(thrust::device_vector<Int>& voTarget, UInt vNewSize, Int vFillData)
{
	voTarget.resize(vNewSize, vFillData);
}

void resizeDeviceVector(thrust::device_vector<Real>& voTarget, UInt vNewSize, Real vFillData)
{
	voTarget.resize(vNewSize, vFillData);
}

void resizeDeviceVector(thrust::device_vector<Vector3>& voTarget, UInt vNewSize, Vector3 vFillData)
{
	voTarget.resize(vNewSize, vFillData);
}

void resizeDeviceVector(thrust::device_vector<bool>& voTarget, UInt vNewSize, bool vFillData)
{
	voTarget.resize(vNewSize, vFillData);
}

void mallocDevicePtrReal(thrust::device_ptr<Real>& voTarget, Real vSize)
{
	voTarget = thrust::device_malloc<Real>(static_cast<size_t>(vSize) * sizeof(Real));
}

void fillDevicePtrReal(thrust::device_ptr<Real>& voTarget, UInt vSize, Real vValue)
{
	thrust::fill(voTarget, voTarget + vSize, vValue);
}

void fillDeviceVectorReal(thrust::device_vector<Real>& voTarget, Real vValue)
{
	thrust::fill(voTarget.begin(), voTarget.end(), vValue);
}

void fillDeviceVectorVector3(thrust::device_vector<Vector3>& voTarget, Vector3 vValue)
{
	thrust::fill(voTarget.begin(), voTarget.end(), vValue);
}

void copyDeviceVectorToPointerReal(const thrust::device_vector<Real>& vSource, thrust::device_ptr<Real>& voTarget)
{
	thrust::copy(vSource.begin(), vSource.end(), voTarget);
}

void sendSTDVectToDeviceReal(const vector<Real>& vSTDVector, thrust::device_vector<Real>& voTarget)
{
	if (voTarget.size() != vSTDVector.size())
	{
		voTarget.resize(vSTDVector.size());
	}
	CHECK_CUDA(cudaMemcpy(
		raw_pointer_cast(voTarget.data()),
		vSTDVector.data(),
		vSTDVector.size() * sizeof(Real),
		cudaMemcpyHostToDevice)
	);
}

void sendSTDVectToDeviceUInt(const vector<UInt>& vSTDVector, thrust::device_vector<UInt>& voTarget)
{
	if (voTarget.size() != vSTDVector.size())
	{
		voTarget.resize(vSTDVector.size());
	}
	CHECK_CUDA(cudaMemcpy(
		raw_pointer_cast(voTarget.data()),
		vSTDVector.data(),
		vSTDVector.size() * sizeof(UInt),
		cudaMemcpyHostToDevice)
	);
}

const UInt* getReadOnlyRawDevicePointer(const thrust::device_vector<UInt>& vTarget)
{
	return raw_pointer_cast(vTarget.data());
}

const Int * getReadOnlyRawDevicePointer(const thrust::device_vector<Int>& vTarget)
{
	return raw_pointer_cast(vTarget.data());
}

const Real * getReadOnlyRawDevicePointer(const thrust::device_vector<Real>& vTarget)
{
	return raw_pointer_cast(vTarget.data());
}

const Vector3 * getReadOnlyRawDevicePointer(const thrust::device_vector<Vector3>& vTarget)
{
	return raw_pointer_cast(vTarget.data());
}

const Real * getReadOnlyRawDevicePointer(const thrust::device_ptr<Real>& vTarget)
{
	return raw_pointer_cast(vTarget);
}

thrust::device_ptr<const Real> getReadOnlyDevicePointer(const thrust::device_vector<Real>& vTarget)
{
	return vTarget.data();
}

thrust::device_ptr<Real> getDevicePointer(thrust::device_vector<Real>& vioTarget)
{
	return vioTarget.data();
}

UInt* getRawDevicePointerUInt(thrust::device_vector<UInt>& vioTarget)
{
	return raw_pointer_cast(vioTarget.data());
}

Int * getRawDevicePointerInt(thrust::device_vector<Int>& vioTarget)
{
	return raw_pointer_cast(vioTarget.data());
}

Real * getRawDevicePointerReal(thrust::device_vector<Real>& vioTarget)
{
	return raw_pointer_cast(vioTarget.data());
}

Real * getRawDevicePointerReal(thrust::device_ptr<Real>& vioTarget)
{
	return raw_pointer_cast(vioTarget);
}

Vector3 * getRawDevicePointerVector3(thrust::device_vector<Vector3>& vioTarget)
{
	return raw_pointer_cast(vioTarget.data());
}

bool * getRawDevicePointerBool(thrust::device_vector<bool>& vioTarget)
{
	return raw_pointer_cast(vioTarget.data());
}

UInt getElementUInt(const thrust::device_vector<UInt>& vTarget, UInt vIndex)
{
	_ASSERT(vIndex < vTarget.size());
	return vTarget[vIndex];
}

Real getElementReal(const thrust::device_vector<Real>& vTarget, UInt vIndex)
{
	_ASSERT(vIndex < vTarget.size());
	return vTarget[vIndex];
}

Real getElementReal(const thrust::device_ptr<Real>& vTarget, UInt vIndex)
{
	_ASSERT(vTarget != nullptr);
	return vTarget[vIndex];
}

Vector3 getElementVector3(const thrust::device_vector<Vector3>& vTarget, UInt vIndex)
{
	_ASSERT(vIndex < vTarget.size());
	return vTarget[vIndex];
}

void setElementReal(thrust::device_vector<Real>& voTarget, UInt vIndex, Real vValue)
{
	voTarget[vIndex] = vValue;
}

void outputPosVector(const thrust::device_vector<Real>& vTarget)
{
	UInt PosSize = static_cast<UInt>(vTarget.size() / 3);
	for (UInt i = 0; i < PosSize; i++)
	{
		cout << "Pos No." << i << ":" << vTarget[i] << "," << vTarget[PosSize + i] << "," << vTarget[PosSize * 2 + i] << endl;
	}
}

void outputUIntVector(const thrust::device_vector<UInt>& vTarget)
{
	for (int i = 0; i < vTarget.size(); i++)
	{
		cout << vTarget[i] << ",";
	}
	cout << endl;
}

struct saxpy_functor
{
	const float a;

	saxpy_functor(float _a) : a(_a) {}

	__host__ __device__
		float operator()(const float& x, const float& y) const {
		return a * x + y;
	}
};

void axpyReal(const thrust::device_ptr<Real>& vX, thrust::device_ptr<Real>& vioY, UInt vSize, Real vFactor)
{
	thrust::transform(vX, vX + vSize, vioY, vioY + vSize, saxpy_functor(vFactor));
}

void axpyReal(const thrust::device_vector<Real>& vX, thrust::device_vector<Real>& vioY, Real vFactor)
{
	CHECK_CUBLAS(cublasAxpy
	(
		CCudaContextManager::getInstance().getCublasHandle(),
		vX.size(),
		&vFactor,
		getReadOnlyRawDevicePointer(vX),
		1,
		getRawDevicePointerReal(vioY),
		1
	));
}

void scaleReal(thrust::device_vector<Real>& vioX, Real vFactor)
{
	CHECK_CUBLAS(cublasScal
	(
		CCudaContextManager::getInstance().getCublasHandle(),
		vioX.size(),
		&vFactor,
		getRawDevicePointerReal(vioX),
		1
	));
}

struct SAccumlateFunc
{
	const Real m_Factor;

	SAccumlateFunc(Real vFactor) : m_Factor(vFactor) {}

	__host__ __device__
		Real operator()(const Real& x, const Real& y) const {
		return y + m_Factor * (x - y);
	}
};

void smoothAccumlate
(
	const thrust::device_vector<Real>& vPrevData,
	Real vSmoothFactor,
	thrust::device_vector<Real>& voData
)
{
	thrust::transform(vPrevData.begin(), vPrevData.end(), voData.begin(), voData.end(), saxpy_functor(vSmoothFactor));
}

Real getMaxValue(const thrust::device_vector<Real>& vSource)
{
	Int ResultIndex = 0;
	CHECK_CUBLAS(cublasMax
	(
		CCudaContextManager::getInstance().getCublasHandle(),
		vSource.size(),
		getReadOnlyRawDevicePointer(vSource),
		1,
		&ResultIndex
	));
	return vSource[ResultIndex - 1];
}

Real getMinValue(const thrust::device_vector<Real>& vSource)
{
	Int ResultIndex = 0;
	CHECK_CUBLAS(cublasMin
	(
		CCudaContextManager::getInstance().getCublasHandle(),
		vSource.size(),
		getReadOnlyRawDevicePointer(vSource),
		1,
		&ResultIndex
	));
	return vSource[ResultIndex - 1];
}
