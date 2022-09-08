#pragma once
#include "Grid.h"

class CVectorGrid : public CGrid
{
public:
	CVectorGrid();
	virtual ~CVectorGrid();

	const thrust::device_vector<Real>& getConstGridDataX() const;
	const thrust::device_vector<Real>& getConstGridDataY() const;
	const thrust::device_vector<Real>& getConstGridDataZ() const;
	thrust::device_vector<Real>& getGridDataX();
	thrust::device_vector<Real>& getGridDataY();
	thrust::device_vector<Real>& getGridDataZ();
	const Real* getConstGridDataXGPUPtr() const;
	const Real* getConstGridDataYGPUPtr() const;
	const Real* getConstGridDataZGPUPtr() const;
	Real* getGridDataXGPUPtr();
	Real* getGridDataYGPUPtr();
	Real* getGridDataZGPUPtr();

	void resizeDataX(Vector3i vResX);
	void resizeDataY(Vector3i vResY);
	void resizeDataZ(Vector3i vResZ);
	void resizeDataX(const Real* vStartPtrX, const Real* vEndPtrX);
	void resizeDataY(const Real* vStartPtrY, const Real* vEndPtrY);
	void resizeDataZ(const Real* vStartPtrZ, const Real* vEndPtrZ);
	void resizeDataX(const thrust::device_vector<Real>& vDeviceVectorX);
	void resizeDataY(const thrust::device_vector<Real>& vDeviceVectorY);
	void resizeDataZ(const thrust::device_vector<Real>& vDeviceVectorZ);

protected:
	thrust::device_vector<Real> m_DataX;
	thrust::device_vector<Real> m_DataY;
	thrust::device_vector<Real> m_DataZ;
};
