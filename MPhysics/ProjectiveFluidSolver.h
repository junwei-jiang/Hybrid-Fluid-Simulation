#pragma once
#include "CuMatVecMultiplier.h"
#include "CuSparsePCG.h"
#include "KNNSearch.h"
#include "BoundaryVolumeMap.h"

class CProjectiveFluidSolver
{
public:
	CProjectiveFluidSolver() = default;
	~CProjectiveFluidSolver();

	void loadConfg(const boost::property_tree::ptree& vSimualtorData);
	void updateTargetParticleGroup(const shared_ptr<CParticleGroup>& vTargetParticle);
	void addRigidBodyBoundary(const shared_ptr<CRigidBodyBoundaryVolumeMap>& vRigidBodyBoundary);

	void doNeighborSearch();
	void projectAndSolveLiner();
	void applyBoundaryeffect();

	UInt getConstraintSize() const;
	shared_ptr<CParticleGroup> getTargetParticleGroup() const;
	shared_ptr<CRigidBodyBoundaryVolumeMap> getBoundary(Int vBoundaryIndex = 0) const;
	void setStiffness(Real vInput);
	void setRestDensity(Real vInput);
	void setDeltaT(Real vInput);
	void setMaxIteration(UInt vInput);
	void setProjectiveMaxIterationNum(UInt vInput);
	void setProjectiveThreshold(Real vInput);
	Real getDeltaT() const;

	Real getStiffness() const;
	Real getRestDensity() const;

	const CCuDenseVector& getVectorb() const;
	UInt getNeighborCount(UInt vIndex) const;

	const CKNNSearch& getNeighborSearch() const;

private:
	Real m_Stiffness = 50000;
	Real m_RestDensity = 1000;
	UInt m_MaxIteration = 10;
	UInt m_ProjectiveMaxIterationNum = 100;
	Real m_ProjectiveThreshold = 1e-14;

	Real m_DeltaT = 0.0;

	EInterpolationMethod m_InterpolationMethod = EXPLICIT_EULER;
	UInt m_ConstraintSize = 0;

	CKNNSearch m_NeighborSearcher;
	CCuSparsePCG m_LinerSolver;

	vector<shared_ptr<CRigidBodyBoundaryVolumeMap>> m_Boundary;
	shared_ptr<CParticleGroup> m_TargetParticleGroup = nullptr;

	//Cache
	thrust::device_vector<Vector3> m_GradC;
	thrust::device_vector<Vector3> m_P;

	//(ParticleSize * 3) X 1£¨≈≈¡–£∫(x0£¨y0£¨z0£¨°≠°≠£¨xn£¨yn£¨zn)
	CCuDenseVector m_Vectorb;
	CCuDenseVector m_Acceleration;
	void __generateAMatrix();
	void __sloveLiner();
};