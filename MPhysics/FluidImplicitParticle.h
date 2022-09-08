#pragma once
#include "AdvectionSolver.h"
#include "EulerParticles.h"

class CFluidImplicitParticle : public CAdvectionSolver
{
	public:
		CFluidImplicitParticle();
		CFluidImplicitParticle
		(
			const CCellCenteredScalarField& vFluidSDF,
			const CCellCenteredScalarField& vSolidSDF,
			UInt vNumOfPerGrid,
			const Vector3i& vResolution,
			const Vector3& vOrigin = Vector3(0, 0, 0),
			const Vector3& vSpacing = Vector3(1, 1, 1),
			Real vCFLNumber = 1.0,
			EPGTransferAlgorithm vPGTransferAlg = EPGTransferAlgorithm::LINEAR
		);
		~CFluidImplicitParticle();

		void advect
		(
			const CCellCenteredScalarField& vInputField,
			const CFaceCenteredVectorField& vVelocityField,
			Real vDeltaT,
			CCellCenteredScalarField& voOutputField,
			EAdvectionAccuracy vEAdvectionAccuracy = EAdvectionAccuracy::RK3,
			const CCellCenteredScalarField& vBoundarysSDF = CCellCenteredScalarField()
		)override;

		void advect
		(
			const CCellCenteredVectorField& vInputField,
			const CFaceCenteredVectorField& vVelocityField,
			Real vDeltaT,
			CCellCenteredVectorField& voOutputField,
			EAdvectionAccuracy vEAdvectionAccuracy = EAdvectionAccuracy::RK3,
			const CCellCenteredScalarField& vBoundarysSDF = CCellCenteredScalarField()
		) override;

		void advect
		(
			const CFaceCenteredVectorField& vInputField,
			const CFaceCenteredVectorField& vVelocityField,
			Real vDeltaT,
			CFaceCenteredVectorField& voOutputField,
			EAdvectionAccuracy vEAdvectionAccuracy = EAdvectionAccuracy::RK3,
			const CCellCenteredScalarField& vBoundarysSDF = CCellCenteredScalarField()
		)override;

		void resizeFluidImplicitParticle
		(
			const CCellCenteredScalarField& vFluidSDF,
			const CCellCenteredScalarField& vSolidSDF,
			UInt vNumOfPerGrid,
			const Vector3i& vResolution,
			const Vector3& vOrigin = Vector3(0, 0, 0),
			const Vector3& vSpacing = Vector3(1, 1, 1),
			Real vCFLNumber = 1.0,
			EPGTransferAlgorithm vPGTransferAlg = EPGTransferAlgorithm::LINEAR
		);

		const CEulerParticles& getEulerParticles() const;

	private:
		CEulerParticles m_EulerParticles;

		Real m_CFLNumber;
		EPGTransferAlgorithm m_PGTransferAlg;

		CCellCenteredScalarField m_CCSWeightField;
		CCellCenteredVectorField m_CCVWeightField;
		CFaceCenteredVectorField m_FCVWeightField;

		CFaceCenteredVectorField m_DeltaGridVelField;
		thrust::device_vector<Real> m_DeltaParticlesVel;

		bool m_ScalarValueFlag = false;
		bool m_VectorValueFlag = false;
		bool m_VelFlag = false;
};

typedef std::shared_ptr<CFluidImplicitParticle> FluidImplicitParticleSolverPtr;