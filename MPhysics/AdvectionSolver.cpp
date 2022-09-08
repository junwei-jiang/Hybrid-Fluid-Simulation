#include "AdvectionSolver.h"

CAdvectionSolver::CAdvectionSolver()
{

}

CAdvectionSolver::~CAdvectionSolver()
{

}

void CAdvectionSolver::resizeAdvectionSolver(const Vector3i& vResolution, ESamplingAlgorithm vSamplingAlg)
{
	m_Resolution = vResolution;
	m_SamplingAlg = vSamplingAlg;

	m_IsInit = true;
}

void CAdvectionSolver::setSamplingAlg(ESamplingAlgorithm vSamplingAlg)
{
	m_SamplingAlg = vSamplingAlg;
}