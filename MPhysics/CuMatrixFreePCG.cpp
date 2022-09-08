#include "CuMatrixFreePCG.h"
#include "ProjectiveFluidSolver.h"
#include "ThrustWapper.cuh"
#include "SimulationConfigManager.h"
#include "CudaContextManager.h"

void CCuMatrixFreePCG::init(UInt vColSize, UInt vRowSize, UInt vNonZeroSize, MatrixVecCGProd vMatrixVecCGProd, MatrixVecCGProd vMInvxCGProdFunc)
{
	m_ColSize = vColSize;
	m_RowSize = vRowSize;
	m_NonZeroSize = vNonZeroSize;
	m_AxCGProdFunc = vMatrixVecCGProd;
	m_MInvxCGProdFunc = vMInvxCGProdFunc;

	m_ZM1.resize(vColSize, 0);
	m_R.resize(vColSize, 0);
	m_P.resize(vColSize, 0);
	m_TempVector.resize(vColSize, 0);
	m_AP.resize(vColSize, 0);
}

bool CCuMatrixFreePCG::solvePCGInvDia(const CCuDenseVector & vVectorb, CCuDenseVector & voX, void* vUserData)
{
	_ASSERT(m_MInvxCGProdFunc != nullptr);
	_ASSERT(m_AxCGProdFunc != nullptr);

	m_R = vVectorb;
	if (sqrt(m_R.norm2()) < m_Threshold)
	{
		voX.setZero();
		return true;
	}

	m_AxCGProdFunc(voX, m_R, vUserData);
	m_R.scale(-1.0);
	m_R += vVectorb;

	m_MInvxCGProdFunc(m_R, m_ZM1, vUserData);
	m_P = m_ZM1;
	UInt ItNum = 0;
	for (UInt i = 0; i < m_IterationNum; i++)
	{
		Real RkDotZk = m_R * m_ZM1;
		m_AxCGProdFunc(m_P, m_AP, vUserData);
		Real PkDotAPk = m_P * m_AP;
		Real Alpha = RkDotZk / PkDotAPk;

		voX.plusAlphaX(m_P, Alpha);
		m_R.plusAlphaX(m_AP, -Alpha);

		Real Error = sqrt(m_R.norm2());
		if (Error < m_Threshold) break;

		m_MInvxCGProdFunc(m_R, m_ZM1, vUserData);

		Real NewRkDotZk = m_R * m_ZM1;
		Real Beta = NewRkDotZk / RkDotZk;

		m_P *= Beta;
		m_P += m_ZM1;
		ItNum++;
	}

	return ItNum <= 1;
}

bool CCuMatrixFreePCG::solveCG(const CCuDenseVector & vVectorb, CCuDenseVector & voX, void* vUserData)
{
	_ASSERT(m_AxCGProdFunc != nullptr);

	m_R = vVectorb;
	if (sqrt(m_R.norm2()) < m_Threshold)
	{
		voX.setZero();
		return true;
	}

	m_AxCGProdFunc(voX, m_R, vUserData);
	m_R.scale(-1.0);
	m_R += vVectorb;

	m_P = m_R; 
	Real Error = 0;
	UInt ItNum = 0;
	for (UInt i = 0; i < m_IterationNum; i++)
	{
		Real RkDotRk = m_R * m_R;
		m_AxCGProdFunc(m_P, m_AP, vUserData);
		Real PkDotAPk = m_P * m_AP;
		Real Alpha = RkDotRk / PkDotAPk;

		voX.plusAlphaX(m_P, Alpha);
		m_R.plusAlphaX(m_AP, -Alpha);

		Error = sqrt(m_R.norm2());
		if (Error < m_Threshold) break;

		Real NewRkDotNewRk = m_R * m_R;
		Real Beta = NewRkDotNewRk / RkDotRk;

		m_P *= Beta;
		m_P += m_R;
		ItNum++;
	}

	return ItNum <= 2;
}

void CCuMatrixFreePCG::setIterationNum(UInt vIterationNum)
{
	m_IterationNum = vIterationNum;
}

void CCuMatrixFreePCG::setThreshold(Real vThreshold)
{
	m_Threshold = vThreshold;
}

Real CCuMatrixFreePCG::getThreshold()
{
	return m_Threshold;
}