#include "Grid.h"

CGrid::CGrid()
{

}

CGrid::~CGrid()
{

}

const Vector3i& CGrid::getResolution() const
{
	return m_Resolution;
}

const Vector3&  CGrid::getOrigin() const
{
	return m_Origin;
}

const Vector3&  CGrid::getSpacing() const
{
	return m_Spacing;
}

void CGrid::setGridParameters(Vector3i vResolution, Vector3 vOrigin, Vector3 vSpacing)
{
	m_Resolution = vResolution;
	m_Origin = vOrigin;
	m_Spacing = vSpacing;
}

void CGrid::setResolution(Vector3i vResolution)
{
	m_Resolution = vResolution;
}

void CGrid::setOrigin(Vector3 vOrigin)
{
	m_Origin = vOrigin;
}

void CGrid::setSpacing(Vector3 vSpacing)
{
	m_Spacing = vSpacing;
}