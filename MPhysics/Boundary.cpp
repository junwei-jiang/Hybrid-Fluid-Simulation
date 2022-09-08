#include "Boundary.h"
#include "BoundaryTool.cuh"

CBoundarys::CBoundarys()
{

}

CBoundarys::CBoundarys(Vector3i vResolution, Vector3 vOrigin, Vector3 vSpacing)
{
	resizeBoundarys(vResolution, vOrigin, vSpacing);
}

void CBoundarys::resizeBoundarys(Vector3i vResolution, Vector3 vOrigin, Vector3 vSpacing)
{
	m_BoundarysResolution = vResolution;
	m_BoundarysOrigin = vOrigin;
	m_BoundarysSpacing = vSpacing;

	m_NumOfBoundarys = 0;

	clearDeviceVectorReal(m_BounadrysVel);
	clearDeviceVectorReal(m_BounadrysTranslation);
	clearDeviceVectorReal(m_BounadrysRotation);
	clearDeviceVectorReal(m_BounadrysScale);

	m_OriginalBoundarysSDF.clear();
	m_CurrentBoundarysSDF.clear();

	m_TotalBoundarysMarker.resize(vResolution, vOrigin, vSpacing);
	m_TotalBoundarysSDF.resize(vResolution, vOrigin, vSpacing);
	m_TotalBoundarysVel.resize(vResolution, vOrigin, vSpacing);

	m_SamplingPosField.resize(vResolution, vOrigin, vSpacing);
}

void CBoundarys::setTotalBoundarysVel(const CFaceCenteredVectorField& vBoundarysVelField)
{
	m_TotalBoundarysVel = vBoundarysVelField;
}

int CBoundarys::getNumOfBoundarys() const
{
	return m_NumOfBoundarys;
}

const thrust::device_vector<Real>& CBoundarys::getBoundarysVel() const
{
	return m_BounadrysVel;
}

const thrust::device_vector<Real>& CBoundarys::getBoundarysTranslation() const
{
	return m_BounadrysTranslation;
}

const thrust::device_vector<Real>& CBoundarys::getBoundarysRotation() const
{
	return m_BounadrysRotation;
}

const thrust::device_vector<Real>& CBoundarys::getBoundarysScale() const
{
	return m_BounadrysScale;
}

const CCellCenteredScalarField& CBoundarys::getOriginalBoundarysSDF(Int vIndex) const
{
	return m_OriginalBoundarysSDF[vIndex];
}

const CCellCenteredScalarField& CBoundarys::getCurrentBoundarysSDF(Int vIndex) const
{
	return m_CurrentBoundarysSDF[vIndex];
}

const CCellCenteredScalarField& CBoundarys::getTotalBoundarysMarker() const
{
	return m_TotalBoundarysMarker;
}

const CCellCenteredScalarField& CBoundarys::getTotalBoundarysSDF() const
{
	return m_TotalBoundarysSDF;
}

const CFaceCenteredVectorField& CBoundarys::getTotalBoundarysVel() const
{
	return m_TotalBoundarysVel;
}

void CBoundarys::addBoundary(const CCellCenteredScalarField& vNewBoundarySDFField, Vector3 vVelocity, Vector3 vTranslation, Vector3 vRotation, Vector3 vScale)
{
	CCellCenteredScalarField TempBoundarySDF = vNewBoundarySDFField;

	m_OriginalBoundarysSDF.push_back(TempBoundarySDF);
	m_CurrentBoundarysSDF.push_back(TempBoundarySDF);

	push_backDeviceVectorReal(m_BounadrysVel, vVelocity.x);
	push_backDeviceVectorReal(m_BounadrysVel, vVelocity.y);
	push_backDeviceVectorReal(m_BounadrysVel, vVelocity.z);
	push_backDeviceVectorReal(m_BounadrysTranslation, vTranslation.x);
	push_backDeviceVectorReal(m_BounadrysTranslation, vTranslation.y);
	push_backDeviceVectorReal(m_BounadrysTranslation, vTranslation.z);
	push_backDeviceVectorReal(m_BounadrysRotation, vRotation.x);
	push_backDeviceVectorReal(m_BounadrysRotation, vRotation.y);
	push_backDeviceVectorReal(m_BounadrysRotation, vRotation.z);
	push_backDeviceVectorReal(m_BounadrysScale, vScale.x);
	push_backDeviceVectorReal(m_BounadrysScale, vScale.y);
	push_backDeviceVectorReal(m_BounadrysScale, vScale.z);

	m_NumOfBoundarys++;
}

void CBoundarys::updateBoundarys(Real vDeltaT)
{
	__moveBoundarys(vDeltaT);
	__resamplingCurrentBoundarysSDF();
	__updateTotalBoundarysMarker();
	__updateTotalBoundarysSDF();
	__updateTotalBoundarysVel();
}

void CBoundarys::__moveBoundarys(Real vDeltaT)
{
	axpyReal(m_BounadrysVel, m_BounadrysTranslation, -vDeltaT);
}

void CBoundarys::__resamplingCurrentBoundarysSDF()
{
	for (int i = 0; i < m_NumOfBoundarys; i++)
	{
		transformFieldInvoker(m_SamplingPosField, m_BounadrysTranslation, m_BounadrysRotation, m_BounadrysScale, i);
		m_OriginalBoundarysSDF[i].sampleField(m_SamplingPosField, m_CurrentBoundarysSDF[i]);
	}
}

void CBoundarys::__updateTotalBoundarysSDF()
{
	buildSolidsMarkerInvoker(m_CurrentBoundarysSDF, m_TotalBoundarysSDF);
}

void CBoundarys::__updateTotalBoundarysMarker()
{
	buildSolidsMarkerInvoker(m_CurrentBoundarysSDF, m_TotalBoundarysMarker);
}

void CBoundarys::__updateTotalBoundarysVel()
{
	buildSolidsVelFieldInvoker(m_TotalBoundarysMarker, m_BounadrysVel, m_TotalBoundarysVel);
}