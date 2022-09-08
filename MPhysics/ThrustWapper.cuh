#pragma once
#include "Common.h"

void clearDeviceVectorReal(thrust::device_vector<Real>& vioSource);

void push_backDeviceVectorReal(thrust::device_vector<Real>& vioSource, Real vValue);

UInt getDeviceVectorSize(const thrust::device_vector<Real>& vSource);
UInt getDeviceVectorSize(const thrust::device_vector<UInt>& vSource);
UInt getDeviceVectorSize(const thrust::device_vector<Vector3>& vSource);
UInt getDeviceVectorSize(const thrust::device_vector<Int>& vSource);

void appendSTDVectorToDeviceVectorReal(const vector<Real>& vSTDVector, thrust::device_vector<Real>& voSource);
void appendDeviceVectorToDeviceVectorReal(const thrust::device_vector<Real>& vDeviceVector, thrust::device_vector<Real>& voSource);

void assignDeviceVectorReal(thrust::device_vector<Real>& voTarget, const Real* vDataStartPtr, const Real* vDataEndPtr);
void assignDeviceVectorInt(thrust::device_vector<Int>& voTarget, const Int* vDataStartPtr, const Int* vDataEndPtr);
void assignDeviceVectorReal(const thrust::device_vector<Real>& vSource, thrust::device_vector<Real>& voTarget);

void resizeDeviceVector(thrust::device_vector<UInt>& voTarget, UInt vNewSize, UInt vFillData = 0);
void resizeDeviceVector(thrust::device_vector<Int>& voTarget, UInt vNewSize, Int vFillData = 0);
void resizeDeviceVector(thrust::device_vector<Real>& voTarget, UInt vNewSize, Real vFillData = 0);
void resizeDeviceVector(thrust::device_vector<Vector3>& voTarget, UInt vNewSize, Vector3 vFillData = Vector3(0, 0, 0));
void resizeDeviceVector(thrust::device_vector<bool>& voTarget, UInt vNewSize, bool vFillData = false);

void mallocDevicePtrReal(thrust::device_ptr<Real>& voTarget, Real vSize);
void fillDevicePtrReal(thrust::device_ptr<Real>& voTarget, UInt vSize, Real vValue);
void fillDeviceVectorReal(thrust::device_vector<Real>& voTarget, Real vValue);
void fillDeviceVectorVector3(thrust::device_vector<Vector3>& voTarget, Vector3 vValue);
void copyDeviceVectorToPointerReal(
	const thrust::device_vector<Real>& vSource,
	thrust::device_ptr<Real>& voTarget
);

void sendSTDVectToDeviceReal(const vector<Real>& vSTDVector, thrust::device_vector<Real>& voTarget);
void sendSTDVectToDeviceUInt(const vector<UInt>& vSTDVector, thrust::device_vector<UInt>& voTarget);

const UInt* getReadOnlyRawDevicePointer(const thrust::device_vector<UInt>& vTarget);
const Int* getReadOnlyRawDevicePointer(const thrust::device_vector<Int>& vTarget);
const Real* getReadOnlyRawDevicePointer(const thrust::device_vector<Real>& vTarget);
const Vector3* getReadOnlyRawDevicePointer(const thrust::device_vector<Vector3>& vTarget);
const Real* getReadOnlyRawDevicePointer(const thrust::device_ptr<Real>& vTarget);

thrust::device_ptr<const Real> getReadOnlyDevicePointer(const thrust::device_vector<Real>& vTarget);
thrust::device_ptr<Real> getDevicePointer(thrust::device_vector<Real>& vioTarget);

UInt* getRawDevicePointerUInt(thrust::device_vector<UInt>& vioTarget);
Int* getRawDevicePointerInt(thrust::device_vector<Int>& vioTarget);
Real* getRawDevicePointerReal(thrust::device_vector<Real>& vioTarget);
Real* getRawDevicePointerReal(thrust::device_ptr<Real>& vioTarget);
Vector3* getRawDevicePointerVector3(thrust::device_vector<Vector3>& vioTarget);
bool* getRawDevicePointerBool(thrust::device_vector<bool>& vioTarget);

UInt getElementUInt(const thrust::device_vector<UInt>& vTarget, UInt vIndex);
Real getElementReal(const thrust::device_vector<Real>& vTarget, UInt vIndex);
Real getElementReal(const thrust::device_ptr<Real>& vTarget, UInt vIndex);
Vector3 getElementVector3(const thrust::device_vector<Vector3>& vTarget, UInt vIndex);

void setElementReal(thrust::device_vector<Real>& voTarget, UInt vIndex, Real vValue);

void outputPosVector(const thrust::device_vector<Real>& vTarget);
void outputUIntVector(const thrust::device_vector<UInt>& vTarget);

void axpyReal(const thrust::device_ptr<Real>& vX, thrust::device_ptr<Real>& vioY, UInt vSize, Real vFactor);
void axpyReal(const thrust::device_vector<Real>& vX, thrust::device_vector<Real>& vioY, Real vFactor);
void scaleReal(thrust::device_vector<Real>& vioX, Real vFactor);
void smoothAccumlate(const thrust::device_vector<Real>& vPrevData, Real vSmoothFactor, thrust::device_vector<Real>& voData);

Real getMaxValue(const thrust::device_vector<Real>& vSource);
Real getMinValue(const thrust::device_vector<Real>& vSource);