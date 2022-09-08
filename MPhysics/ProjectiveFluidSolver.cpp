#include "ProjectiveFluidSolver.h"
#include "DensityConstraintKernel.cuh"

#include "ThrustWapper.cuh"
#include "SimulationConfigManager.h"
#include "SPHKernelFunc.cuh"

CProjectiveFluidSolver::~CProjectiveFluidSolver(){}

void CProjectiveFluidSolver::loadConfg(const boost::property_tree::ptree& vSimualtorData)
{
	m_MaxIteration = vSimualtorData.get_child("PFMaxIterationCount").get_value<UInt>();
	m_ProjectiveMaxIterationNum = vSimualtorData.get_child("PFProjectMaxIterationCount").get_value<UInt>();
	m_ProjectiveThreshold = vSimualtorData.get_child("PFProjectErrorThreshold").get_value<Real>();
}

void CProjectiveFluidSolver::updateTargetParticleGroup(const shared_ptr<CParticleGroup>& vTargetParticle)
{
	_ASSERT(vTargetParticle->getSize() > 0);

	m_ConstraintSize = vTargetParticle->getSize();
	m_Vectorb.resize(vTargetParticle->getSize() * 3);
	m_Acceleration.resize(vTargetParticle->getSize() * 3, 0);

	m_TargetParticleGroup = vTargetParticle;

	m_NeighborSearcher.bindParticleGroup(vTargetParticle);
	m_LinerSolver.init(vTargetParticle->getSize(), m_ConstraintSize);

	CCubicKernel SmoothKernelCPU;
	SmoothKernelCPU.setRadius(m_TargetParticleGroup->getParticleSupportRadius());
	initProjectiveSmoothKernelCubic(SmoothKernelCPU);
}

void CProjectiveFluidSolver::addRigidBodyBoundary(const shared_ptr<CRigidBodyBoundaryVolumeMap>& vRigidBodyBoundary)
{
	m_Boundary.push_back(vRigidBodyBoundary);
}

void CProjectiveFluidSolver::doNeighborSearch()
{
	m_NeighborSearcher.search();
}

void CProjectiveFluidSolver::projectAndSolveLiner()
{
	applyBoundaryeffect();

	for (int i = 0; i < m_MaxIteration; i++)
	{
		m_Vectorb.setZero();

		solveDensityConstraintsInvoker(
			m_TargetParticleGroup->getConstParticlePosGPUPtr(),
			getReadOnlyRawDevicePointer(m_NeighborSearcher.getNeighorData()),
			getReadOnlyRawDevicePointer(m_NeighborSearcher.getNeighborCounts()),
			getReadOnlyRawDevicePointer(m_NeighborSearcher.getNeighorOffsets()),
			m_Boundary,
			m_ConstraintSize,
			m_TargetParticleGroup->getParticleVolume(),
			m_Stiffness,
			m_DeltaT,
			m_ProjectiveMaxIterationNum,
			m_ProjectiveThreshold,
			m_Vectorb
		);

		Real Mass = (m_TargetParticleGroup->getParticleVolume() * m_RestDensity);
		m_Vectorb.plusAlphaX(m_TargetParticleGroup->getParticlePos(), Mass);

		__sloveLiner();
	}
}

void CProjectiveFluidSolver::applyBoundaryeffect()
{
	for (UInt i = 0; i < m_Boundary.size(); i++)
	{
		m_Boundary[i]->doInfluenceToParticle(*m_TargetParticleGroup);
	}
}

UInt CProjectiveFluidSolver::getConstraintSize() const
{
	return m_ConstraintSize;
}

shared_ptr<CParticleGroup> CProjectiveFluidSolver::getTargetParticleGroup() const
{
	return m_TargetParticleGroup;
}

shared_ptr<CRigidBodyBoundaryVolumeMap> CProjectiveFluidSolver::getBoundary(Int vBoundaryIndex) const
{
	return m_Boundary[vBoundaryIndex];
}

void CProjectiveFluidSolver::setStiffness(Real vInput)
{
	m_Stiffness = vInput;
}

void CProjectiveFluidSolver::setRestDensity(Real vInput)
{
	m_RestDensity = vInput;
}

void CProjectiveFluidSolver::setDeltaT(Real vInput)
{
	m_DeltaT = vInput;
}

void CProjectiveFluidSolver::setMaxIteration(UInt vInput)
{
	m_MaxIteration = vInput;
}

void CProjectiveFluidSolver::setProjectiveMaxIterationNum(UInt vInput)
{
	m_ProjectiveMaxIterationNum = vInput;
}
void CProjectiveFluidSolver::setProjectiveThreshold(Real vInput)
{
	m_ProjectiveThreshold = vInput;
}

Real CProjectiveFluidSolver::getDeltaT() const
{
	return m_DeltaT;
}

Real CProjectiveFluidSolver::getStiffness() const
{
	return m_Stiffness;
}

Real CProjectiveFluidSolver::getRestDensity() const
{
	return m_RestDensity;
}

const CCuDenseVector & CProjectiveFluidSolver::getVectorb() const
{
	return m_Vectorb;
}

UInt CProjectiveFluidSolver::getNeighborCount(UInt vIndex) const
{
	return m_NeighborSearcher.getNeighborCount(vIndex);
}

const CKNNSearch & CProjectiveFluidSolver::getNeighborSearch() const
{
	return m_NeighborSearcher;
}

void CProjectiveFluidSolver::__generateAMatrix()
{
}

void CProjectiveFluidSolver::__sloveLiner()
{
	solveLinerInvoker
	(
		m_Vectorb.getConstVectorValue(),
		m_NeighborSearcher.getNeighborCounts(),
		m_DeltaT * m_DeltaT,
		m_Stiffness,
		m_TargetParticleGroup->getParticleVolume() * m_RestDensity,
		m_TargetParticleGroup->getParticlePos().getVectorValue()
	);
}
