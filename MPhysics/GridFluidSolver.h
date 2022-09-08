#pragma once
#include "FieldMathTool.cuh"
#include "EulerSolverTool.cuh"
#include "EulerParticlesTool.cuh"
#include "HybridSimulatorKernel.cuh"
#include "AdvectionSolver.h"
#include "SemiLagrangian.h"
#include "ParticleInCell.h"
#include "FluidImplicitParticle.h"
#include "MixPICAndFLIP.h"
#include "ExternalForcesSolver.h"
#include "PressureSolver.h"
#include "Boundary.h"
#include "BoundaryHelper.h"
#include "CubicLagrangeDiscreteGrid.h"

class CGridFluidSolver
{
public:
	CGridFluidSolver();
	CGridFluidSolver
	(
		Real vDeltaT, 
		const Vector3i& vResolution, 
		const Vector3& vOrigin = Vector3(0, 0, 0), 
		const Vector3& vSpacing = Vector3(1, 1, 1)
	);
	~CGridFluidSolver();

	void resizeGridFluidSolver(const boost::property_tree::ptree& vGridSolverData);
	void generateFluid(const boost::property_tree::ptree& vGridSolverData);
	void resizeGridFluidSolver
	(
		Real vDeltaT,
		const Vector3i& vResolution,
		const Vector3& vOrigin = Vector3(0, 0, 0),
		const Vector3& vSpacing = Vector3(1, 1, 1)
	);

	void addSolidBoundary(const CCellCenteredScalarField& vSolidSDFField);
	void setSolidVelField(const CFaceCenteredVectorField& vSolidVelField);

	void addExternalForce(Vector3 vExternalForce);

	void setColorField(const CCellCenteredVectorField& vColorField);
	void setFluidDomainField(const CCellCenteredScalarField& vFluidDomainField);
	void setVelocityField(const CFaceCenteredVectorField& vVelocityField);

	void setAdvectionSolver(const AdvectionSolverPtr& vAdvectionSolverPtr);
	void setExternalForcesSolver(const ExternalForcesSolverPtr& vExternalForcesSolverPtr);
	void setPressureSolver(const PressureSolverPtr& vPressureSolverPtr);

	const CCellCenteredScalarField& getSolidSDFField() const;

	Real getCurSimulationTime() const;
	const CCellCenteredVectorField& getColorField() const;
	const CCellCenteredScalarField& getFluidDomainField() const;
	const CCellCenteredScalarField& getFluidDomainFieldBeforePressure() const;
	const CCellCenteredScalarField& getFluidSDFField() const;
	const CCellCenteredScalarField& getFluidDensityField() const;
	const CFaceCenteredVectorField& getVelocityField() const;
	const CFaceCenteredVectorField& getVelocityFieldBeforePressure() const;
	const AdvectionSolverPtr& getAdvectionSolver() const;
	const PressureSolverPtr& getPressureSolver() const;

	void update();

	void updateWithoutPressure();
	void solvePressure();

	void generateFluidDomainFromBBox(Vector3 vMin, Vector3 vMax);
	void generateSolidSDFFromCLDGrid(const std::shared_ptr<CCubicLagrangeDiscreteGrid>& vCLDGrid, Vector3 vVel = Vector3(0, 0, 0));
	void generateSolidSDFFromOBJFile(string vTriangleMeshFilePath, bool vIsInvSign = true);

	void generateCurFluidSDF();
	void generateCurFluidDensity();

	void mixVelFieldWithOthers
	(
		const CFaceCenteredVectorField& vOtherVelField,
		const CCellCenteredScalarField& vOthersDensityField,
		const CCellCenteredScalarField& vFluidDensityField
	);

	void transferFluidVelField2Particles
	(	
		const thrust::device_vector<Real>& vParticlesPos,
		thrust::device_vector<Real>& voParticlesVel,
		EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR
	);

protected:
	void _onAdvanceTimeStep(Real vDeltaT);

	void _onBeginAdvanceTimeStep(Real vDeltaT);

	void _onEndAdvanceTimeStep(Real vDeltaT);

	void _computeAdvection(Real vDeltaT);

	void _computeExternalForces(Real vDeltaT);

	void _computeViscosity(Real vDeltaT);

	void _computePressure(Real vDeltaT);

	void _extrapolatingVel();

	void _fixFluidDomain();

private:
	Vector3i m_Resolution;
	Vector3 m_Origin;
	Vector3 m_Spacing;

	Real m_DeltaT;
	Real m_CurSimulationTime = 0.0;
	UInt m_ExtrapolatingNums = 1000;
	bool m_BufferFlag = true;

	CBoundarys m_Boundarys;

	CCellCenteredVectorField m_ColorField1;
	CCellCenteredVectorField m_ColorField2;
	CCellCenteredScalarField m_FluidDomainField1;
	CCellCenteredScalarField m_FluidDomainField2;
	CCellCenteredScalarField m_FluidSDFField;
	CCellCenteredScalarField m_FluidDensityField;
	CFaceCenteredVectorField m_VelocityField1;
	CFaceCenteredVectorField m_VelocityField2;

	CCellCenteredVectorField m_SamplingPosField;
	CFaceCenteredVectorField m_ExtrapolatingVelMarkersField;

	AdvectionSolverPtr m_AdvectionSolver;
	ExternalForcesSolverPtr m_ExternalForcesSolver;
	PressureSolverPtr m_PressureSolver;
};