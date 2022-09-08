#pragma once
#include "Common.h"
#include "ThrustWapper.cuh"

class CGrid
{
public:
	CGrid();
	virtual ~CGrid();

	const Vector3i& getResolution() const;
	const Vector3&  getOrigin() const;
	const Vector3&  getSpacing() const;

	void setGridParameters(Vector3i vResolution, Vector3 vOrigin = Vector3(0, 0, 0), Vector3 vSpacing = Vector3(1, 1, 1));
	void setResolution(Vector3i vResolution);
	void setOrigin(Vector3 vOrigin);
	void setSpacing(Vector3 vSpacing);

private:
	Vector3i m_Resolution;
	Vector3  m_Origin = Vector3(0, 0, 0);
	Vector3  m_Spacing = Vector3(1, 1, 1);
};

