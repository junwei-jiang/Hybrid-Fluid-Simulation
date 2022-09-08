#include "ExternalForcesSolver.h"

CExternalForcesSolver::CExternalForcesSolver()
{

}

CExternalForcesSolver::~CExternalForcesSolver()
{

}

CExternalForcesSolver::CExternalForcesSolver(const Vector3i& vResolution, const Vector3& vOrigin, const Vector3& vSpacing)
{
	resizeExternalForcesSolver(vResolution, vOrigin, vSpacing);
}

void CExternalForcesSolver::resizeExternalForcesSolver(const Vector3i& vResolution, const Vector3& vOrigin, const Vector3& vSpacing)
{
	m_ExternalForces = Vector3(0, 0, 0);


	Vector3i ResX = vResolution + Vector3i(1, 0, 0);
	Vector3i ResY = vResolution + Vector3i(0, 1, 0);
	Vector3i ResZ = vResolution + Vector3i(0, 0, 1);

	vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z, 1);
	vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z, 1);
	vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 1);

	//for (int z = 0; z < ResY.z; z++)
	//{
	//	for (int y = 0; y < ResY.y; y++)
	//	{
	//		for (int x = 0; x < ResY.x; x++)
	//		{
	//			SrcVectorFieldDataY[z * ResY.x * ResY.y + y * ResY.x + x] = y;
	//		}
	//	}
	//}

	m_UnitFieldFCV.resize(vResolution, vOrigin, vSpacing, SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
}

void CExternalForcesSolver::addExternalForces(Vector3 vExternalForces)
{
	m_ExternalForces += vExternalForces;
}

void CExternalForcesSolver::applyExternalForces
(
	CFaceCenteredVectorField& vioVelField,
	Real vDeltaT,
	const CCellCenteredScalarField& vBoundarySDF
)
{
	vioVelField.plusAlphaX(m_UnitFieldFCV, m_Gravity * vDeltaT);
	vioVelField.plusAlphaX(m_UnitFieldFCV, m_ExternalForces * vDeltaT);
}