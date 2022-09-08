#pragma once
#include "Grid.h"

class CScalarGrid : public CGrid
{
public:
	CScalarGrid();
	virtual ~CScalarGrid();

	const thrust::device_vector<Real>& getConstGridData() const;
	thrust::device_vector<Real>& getGridData();
	const Real* getConstGridDataGPUPtr() const;
	Real* getGridDataGPUPtr();

	void resizeData(Vector3i vRes);
	void resizeData(const Real* vStartPtr, const Real* vEndPtr);
	void resizeData(const thrust::device_vector<Real>& vDeviceVector);

protected:
	thrust::device_vector<Real> m_Data;
};