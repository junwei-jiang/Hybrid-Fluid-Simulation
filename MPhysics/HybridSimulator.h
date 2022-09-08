#pragma once
#include "Common.h"
#include "ProjectiveFluidSolver.h"
#include "GridFluidSolver.h"
#include "Particle.h"
#include "CubicLagrangeDiscreteGrid.h"
#include "CellCenteredScalarField.h"

class CHybridSimulator
{
	public:
		CHybridSimulator(const boost::property_tree::ptree& vSimualtorData);
		void update(Real vRealwroldDeltaTime);

		void transformRigidBoundary
		(
			UInt vStaticBoundaryIndex,
			UInt vInstanceIndex,
			SMatrix3x3 vRotation,
			Vector3 vPos
		);	
		UInt addRigidBoundaryInstance
		(
			UInt vStaticBoundaryIndex,
			SMatrix3x3 vRotation,
			Vector3 vPos
		);

		const std::shared_ptr<CParticleGroup>& getTargetParticleGroup() const;
		const CEulerParticles& getEulerParticles() const;
		const CCellCenteredScalarField& getFluidSDF() const;
		UInt getConstNeighborDataSize() const;
		const UInt* getConstNeighborDataGPUPtr() const;
		const UInt* getConstNeighborCountGPUPtr() const;
		const UInt* getConstNeighborOffsetGPUPtr() const;

		Vector3 getGridOrigin() const;
		Vector3 getGridSpacing() const;
		SGridInfo getGridInfo() const;
		Real getCellSize() const;
		UInt getCellDataSize() const;
		const UInt* getConstCellParticleCountsGPUPtr() const;
		const UInt* getConstCellParticleOffsetsGPUPtr() const;

	private:
		Real m_AccumulateTime = 0.0;
		Real m_DistanceDamp = 2.0 / 3.0;
		UInt m_MixSubstepCount = 4;
		Real m_GRate = 1.0;
		Real m_GFRate = 2.0;
		Real m_FRate = 4.0;
		Real m_PRate = 6.0;
		Real m_DeleteRate = 6.0;
		Real m_TimeDeleteRate = 10.0;
		Real m_CreateRate = 3.0;
		Real m_RhoMin = 0.55;
		UInt m_SampleK = 32;
		UInt m_LiveRate = 10;
		UInt m_MaxParticleCount = 100000;

		UInt m_UpdateSteps = 0;

		Vector3i m_GridResolution;
		Vector3 m_GridMin;
		Vector3 m_GridSpace;

		vector<std::shared_ptr<CRigidBodyBoundaryVolumeMap>> m_RigidBoundary;
		std::shared_ptr<CParticleGroup> m_Particles = nullptr;

		CProjectiveFluidSolver m_ProjectiveFluidSolver;
		CGridFluidSolver m_GridFluidSolver;

		CCellCenteredScalarField m_RhoGridG;//δ��ϵ������ܶ�
		CCellCenteredScalarField m_RhoGridP;//���ӹ�դ������ӷ��ܶ�
		CCellCenteredScalarField m_RhoGridC;//�ѻ�ϵ������ܶ�

		CFaceCenteredVectorField m_ParticlesVelField;

		CCellCenteredScalarField m_DistGridG;//���������δ��ϵ�Һ��ľ���
		CCellCenteredScalarField m_DistGridC;//����������ѻ�ϵ�Һ��ľ���
		CCellCenteredScalarField m_DistGridInside;//�����ڲ����������δ��ϵ�Һ��ľ���

		thrust::device_vector<Real> m_ParticlesMass;

		thrust::device_vector<Real> m_DistPG;//���ӵ�GҺ��ľ���
		thrust::device_vector<Real> m_DistPGCache;//���ӵ�Һ��ľ����ʱ�����õĻ���

		thrust::device_vector<Real> m_DistPC;//���ӵ�CҺ��ľ���
		thrust::device_vector<Real> m_DistPInside;//���ӵ�GҺ����ڲ�����
		thrust::device_vector<bool> m_Filter;//���ӵ�ɾ����־
		thrust::device_vector<Real> m_PICVel;//���ӵ�Particle in Cell�ٶ�
		thrust::device_vector<Real> m_FLIPVel;//���ӵ�FLIP�ٶ�

		CFaceCenteredVectorField m_DeltaGridVelField;
		thrust::device_vector<Real> m_DeltaParticlesVel;

		thrust::device_vector<UInt> m_NeedGenNewParticleCellIndexCache;

		CCellCenteredScalarField m_CCSWeightField;
		CFaceCenteredVectorField m_FCVWeightField;

		CCuDenseVector m_PredictVel;
		CCuDenseVector m_PredictPos;

		void __keepParticleTypeSurface();
		void __applyParticleInfluenceToGrid();
		void __applyGridInfluenceToParticle();
		void __addBoundary
		(
			const vector<std::string> & vTriangleMeshFilePath,
			const vector<SBoundaryTransform> & vBoundaryInitTransform,
			Vector3ui vRes, 
			Real vExtension, 
			bool isInv, 
			bool vIsDynamic
		);
		void __resizeParticlesCache();
		void __generateTempGridData();
		void __predict(Real vDeltaTime);
};