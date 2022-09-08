#include "GridFluidSolver.h"

EPGTransferAlgorithm transferString2PGTransferAlg(string vString)
{
	if (vString == "P2GSUM")
	{
		return EPGTransferAlgorithm::P2GSUM;
	}
	else if (vString == "G2PNEAREST")
	{
		return EPGTransferAlgorithm::G2PNEAREST;
	}
	else if (vString == "QUADRATIC")
	{
		return EPGTransferAlgorithm::QUADRATIC;
	}
	else if (vString == "CUBIC")
	{
		return EPGTransferAlgorithm::CUBIC;
	}
	else
	{
		return EPGTransferAlgorithm::LINEAR;
	}
}

CGridFluidSolver::CGridFluidSolver()
{

}

CGridFluidSolver::CGridFluidSolver
(
	Real vDeltaT, 
	const Vector3i& vResolution, 
	const Vector3& vOrigin, 
	const Vector3& vSpacing
)
{
	resizeGridFluidSolver(vDeltaT, vResolution, vOrigin, vSpacing);
}

CGridFluidSolver::~CGridFluidSolver()
{

}

void CGridFluidSolver::resizeGridFluidSolver(const boost::property_tree::ptree& vGridSolverData)
{
	m_Resolution.x = vGridSolverData.get_child("GridResolution.x").get_value<Real>();
	m_Resolution.y = vGridSolverData.get_child("GridResolution.y").get_value<Real>();
	m_Resolution.z = vGridSolverData.get_child("GridResolution.z").get_value<Real>();

	Vector3 SimulationDomainStart;
	SimulationDomainStart.x = vGridSolverData.get_child("SimulationDomain.Start.x").get_value<Real>();
	SimulationDomainStart.y = vGridSolverData.get_child("SimulationDomain.Start.y").get_value<Real>();
	SimulationDomainStart.z = vGridSolverData.get_child("SimulationDomain.Start.z").get_value<Real>();

	Vector3 SimulationDomainEnd;
	SimulationDomainEnd.x = vGridSolverData.get_child("SimulationDomain.End.x").get_value<Real>();
	SimulationDomainEnd.y = vGridSolverData.get_child("SimulationDomain.End.y").get_value<Real>();
	SimulationDomainEnd.z = vGridSolverData.get_child("SimulationDomain.End.z").get_value<Real>();

	Vector3 SimulationDomainRange = SimulationDomainEnd - SimulationDomainStart;
	m_Spacing = Vector3(SimulationDomainRange.x / m_Resolution.x, SimulationDomainRange.y / m_Resolution.y, SimulationDomainRange.z / m_Resolution.z);

	m_Origin = SimulationDomainStart - m_Spacing;
	m_Resolution += Vector3i(2, 2, 2);

	Real TempDeltaT = vGridSolverData.get_child("DeltaT").get_value<Real>();

	resizeGridFluidSolver(TempDeltaT, m_Resolution, m_Origin, m_Spacing);
}

void CGridFluidSolver::generateFluid(const boost::property_tree::ptree& vGridSolverData)
{
	for (auto EulerFluidDomain : vGridSolverData.get_child("EulerFluidDomain"))
	{
		SAABB EulerFluidDomainBBOX;
		EulerFluidDomainBBOX.Min.x = EulerFluidDomain.second.get_child("Range.Start.x").get_value<Real>();
		EulerFluidDomainBBOX.Min.y = EulerFluidDomain.second.get_child("Range.Start.y").get_value<Real>();
		EulerFluidDomainBBOX.Min.z = EulerFluidDomain.second.get_child("Range.Start.z").get_value<Real>();
		EulerFluidDomainBBOX.Max.x = EulerFluidDomain.second.get_child("Range.End.x").get_value<Real>();
		EulerFluidDomainBBOX.Max.y = EulerFluidDomain.second.get_child("Range.End.y").get_value<Real>();
		EulerFluidDomainBBOX.Max.z = EulerFluidDomain.second.get_child("Range.End.z").get_value<Real>();

		generateFluidDomainFromBBox(EulerFluidDomainBBOX.Min, EulerFluidDomainBBOX.Max);
	}

	if (vGridSolverData.get_child("AdvectionSolver").get_child("Type").get_value<string>() == "SemiLagrangian")
	{

	}
	else
	{
		string AdvectionSolverType = vGridSolverData.get_child("AdvectionSolver").get_child("Type").get_value<string>();
		UInt NumOfParticlesPerCell = vGridSolverData.get_child("AdvectionSolver").get_child("NumOfParticlesPerCell").get_value<UInt>();
		Real CFLNumber = vGridSolverData.get_child("AdvectionSolver").get_child("CFLNumber").get_value<Real>();
		EPGTransferAlgorithm PGTransferAlg = transferString2PGTransferAlg(vGridSolverData.get_child("AdvectionSolver").get_child("PGTransferAlg").get_value<string>());

		if (AdvectionSolverType == "PIC")
		{
			ParticleInCellSolverPtr PICSolverPtr = make_shared<CParticleInCell>();
			PICSolverPtr->resizeParticleInCell
			(
				getFluidDomainField(),
				getSolidSDFField(),
				NumOfParticlesPerCell,
				m_Resolution,
				m_Origin,
				m_Spacing,
				CFLNumber,
				PGTransferAlg
			);
			setAdvectionSolver(PICSolverPtr);
		}
		else if (AdvectionSolverType == "FLIP")
		{
			FluidImplicitParticleSolverPtr FLIPSolverPtr = make_shared<CFluidImplicitParticle>();

			FLIPSolverPtr->resizeFluidImplicitParticle
			(
				getFluidDomainField(),
				getSolidSDFField(),
				NumOfParticlesPerCell,
				m_Resolution,
				m_Origin,
				m_Spacing,
				CFLNumber,
				PGTransferAlg
			);
			setAdvectionSolver(FLIPSolverPtr);
		}
		else if (AdvectionSolverType == "MixPICAndFLIP")
		{
			Real MixingCoefficient = vGridSolverData.get_child("AdvectionSolver").get_child("MixingCoefficient").get_value<Real>();

			MixPICAndFLIPSolverPtr MixPICAndFLIPSolverPtr = make_shared<CMixPICAndFLIP>();

			MixPICAndFLIPSolverPtr->resizeMixPICAndFLIP
			(
				getFluidDomainField(),
				getSolidSDFField(),
				NumOfParticlesPerCell,
				m_Resolution,
				m_Origin,
				m_Spacing,
				CFLNumber,
				MixingCoefficient,
				PGTransferAlg
			);
			setAdvectionSolver(MixPICAndFLIPSolverPtr);
		}
		else
		{

		}
	}
}

void CGridFluidSolver::resizeGridFluidSolver
(
	Real vDeltaT, 
	const Vector3i& vResolution, 
	const Vector3& vOrigin, 
	const Vector3& vSpacing
)
{
	m_DeltaT = vDeltaT;

	m_Resolution = vResolution;
	m_Origin = vOrigin;
	m_Spacing = vSpacing;

	m_CurSimulationTime = 0.0;
	m_ExtrapolatingNums = 1000;
	m_BufferFlag = true;

	m_Boundarys.resizeBoundarys(vResolution, vOrigin, vSpacing);

	m_ColorField1.resize(vResolution, vOrigin, vSpacing);
	m_ColorField2.resize(vResolution, vOrigin, vSpacing);
	m_FluidDomainField1.resize(vResolution, vOrigin, vSpacing);
	m_FluidDomainField2.resize(vResolution, vOrigin, vSpacing);
	m_FluidSDFField.resize(vResolution, vOrigin, vSpacing);
	m_FluidDensityField.resize(vResolution, vOrigin, vSpacing);
	m_VelocityField1.resize(vResolution, vOrigin, vSpacing);
	m_VelocityField2.resize(vResolution, vOrigin, vSpacing); 

	m_SamplingPosField.resize(vResolution, vOrigin, vSpacing);
	m_ExtrapolatingVelMarkersField.resize(vResolution, vOrigin, vSpacing);

	m_AdvectionSolver = make_shared<CSemiLagrangian>(vResolution, vOrigin, vSpacing);
	m_ExternalForcesSolver = make_shared<CExternalForcesSolver>(vResolution, vOrigin, vSpacing);
	m_PressureSolver = make_shared<CPressureSolver>(vResolution, vOrigin, vSpacing);
}

void CGridFluidSolver::addExternalForce(Vector3 vExternalForce)
{
	m_ExternalForcesSolver->addExternalForces(vExternalForce);
}

void CGridFluidSolver::addSolidBoundary(const CCellCenteredScalarField& vSolidSDFField)
{
	m_Boundarys.addBoundary(vSolidSDFField);
}

void CGridFluidSolver::setColorField(const CCellCenteredVectorField& vColorField)
{
	m_ColorField1 = vColorField;
	m_ColorField2 = vColorField;
}

void CGridFluidSolver::setFluidDomainField(const CCellCenteredScalarField& vFluidDomainField)
{
	m_FluidDomainField1 = vFluidDomainField;
	m_FluidDomainField2 = vFluidDomainField;
}

void CGridFluidSolver::setSolidVelField(const CFaceCenteredVectorField& vSolidVelField)
{
	m_Boundarys.setTotalBoundarysVel(vSolidVelField);
}

void CGridFluidSolver::setVelocityField(const CFaceCenteredVectorField& vVelocityField)
{
	m_VelocityField1 = vVelocityField;
	m_VelocityField2 = vVelocityField;
}

void CGridFluidSolver::setAdvectionSolver(const AdvectionSolverPtr& vAdvectionSolverPtr)
{
	m_AdvectionSolver = vAdvectionSolverPtr;
}

void CGridFluidSolver::setExternalForcesSolver(const ExternalForcesSolverPtr& vExternalForcesSolverPtr)
{
	m_ExternalForcesSolver = vExternalForcesSolverPtr;
}

void CGridFluidSolver::setPressureSolver(const PressureSolverPtr& vPressureSolverPtr)
{
	m_PressureSolver = vPressureSolverPtr;
}

Real CGridFluidSolver::getCurSimulationTime() const
{
	return m_CurSimulationTime;
}

const CCellCenteredVectorField& CGridFluidSolver::getColorField() const
{
	if (m_BufferFlag)
	{
		return m_ColorField2;
	}
	else
	{
		return m_ColorField1;
	}
}

const CCellCenteredScalarField& CGridFluidSolver::getFluidDomainField() const
{
	if (m_BufferFlag)
	{
		return m_FluidDomainField2;
	}
	else
	{
		return m_FluidDomainField1;
	}
}

const CCellCenteredScalarField& CGridFluidSolver::getFluidDomainFieldBeforePressure() const
{
	if (m_BufferFlag)
	{
		return m_FluidDomainField1;
	}
	else
	{
		return m_FluidDomainField2;
	}
}

const CCellCenteredScalarField& CGridFluidSolver::getFluidSDFField() const
{
	return m_FluidSDFField;
}

const CCellCenteredScalarField& CGridFluidSolver::getSolidSDFField() const
{
	return m_Boundarys.getTotalBoundarysSDF();
}

const CCellCenteredScalarField& CGridFluidSolver::getFluidDensityField() const
{
	return m_FluidDensityField;
}

const CFaceCenteredVectorField& CGridFluidSolver::getVelocityField() const
{
	if (m_BufferFlag)
	{
		return m_VelocityField2;
	}
	else
	{
		return m_VelocityField1;
	}
}

const CFaceCenteredVectorField& CGridFluidSolver::getVelocityFieldBeforePressure() const
{
	if (m_BufferFlag)
	{
		return m_VelocityField1;
	}
	else
	{
		return m_VelocityField2;
	}
}

const AdvectionSolverPtr& CGridFluidSolver::getAdvectionSolver() const
{
	return m_AdvectionSolver;
}

const PressureSolverPtr& CGridFluidSolver::getPressureSolver() const
{
	return m_PressureSolver;
}

void CGridFluidSolver::update()
{
	_onAdvanceTimeStep(m_DeltaT);
}

void CGridFluidSolver::updateWithoutPressure()
{
	_onBeginAdvanceTimeStep(m_DeltaT);

	_computeAdvection(m_DeltaT);
	_computeExternalForces(m_DeltaT);
	_computeViscosity(m_DeltaT);
	//_fixFluidDomain();
}

void CGridFluidSolver::solvePressure()
{
	_computePressure(m_DeltaT);
	_extrapolatingVel();

	_onEndAdvanceTimeStep(m_DeltaT);
}

void CGridFluidSolver::_onAdvanceTimeStep(Real vDeltaT)
{
	_onBeginAdvanceTimeStep(vDeltaT);

	_computeAdvection(vDeltaT);
	_computeExternalForces(vDeltaT);
	_computeViscosity(vDeltaT);
	_computePressure(vDeltaT);
	_extrapolatingVel();

	_onEndAdvanceTimeStep(vDeltaT);
}

void CGridFluidSolver::_onBeginAdvanceTimeStep(Real vDeltaT)
{
	if (m_AdvectionSolver == nullptr)
	{
		std::cout << "Advection solver is nullptr !" << std::endl;
		exit(1);
	}
	if (m_ExternalForcesSolver == nullptr)
	{
		std::cout << "External forces solver is nullptr !" << std::endl;
		exit(1);
	}
	if (m_PressureSolver == nullptr)
	{
		std::cout << "Pressure solver is nullptr !" << std::endl;
		exit(1);
	}
	
	m_Boundarys.updateBoundarys(vDeltaT);
}

void CGridFluidSolver::_onEndAdvanceTimeStep(Real vDeltaT)
{
	m_BufferFlag = !m_BufferFlag;
	m_CurSimulationTime += vDeltaT;
}

void CGridFluidSolver::_computeAdvection(Real vDeltaT)
{
	if (m_BufferFlag)
	{
		//当前时间步处理场缓冲一
		m_AdvectionSolver->advect(m_VelocityField2, m_VelocityField2, vDeltaT, m_VelocityField1, EAdvectionAccuracy::RK3, m_Boundarys.getTotalBoundarysSDF());
		m_AdvectionSolver->advect(m_ColorField2, m_VelocityField2, vDeltaT, m_ColorField1, EAdvectionAccuracy::RK3, m_Boundarys.getTotalBoundarysSDF());
		m_AdvectionSolver->advect(m_FluidDomainField2, m_VelocityField2, vDeltaT, m_FluidDomainField1, EAdvectionAccuracy::RK3, m_Boundarys.getTotalBoundarysSDF());

	}
	else
	{
		//当前时间步处理场缓冲二
		m_AdvectionSolver->advect(m_VelocityField1, m_VelocityField1, vDeltaT, m_VelocityField2, EAdvectionAccuracy::RK3, m_Boundarys.getTotalBoundarysSDF());
		m_AdvectionSolver->advect(m_ColorField1, m_VelocityField1, vDeltaT, m_ColorField2, EAdvectionAccuracy::RK3, m_Boundarys.getTotalBoundarysSDF());
		m_AdvectionSolver->advect(m_FluidDomainField1, m_VelocityField1, vDeltaT, m_FluidDomainField2, EAdvectionAccuracy::RK3, m_Boundarys.getTotalBoundarysSDF());
	}
}

void CGridFluidSolver::_computeExternalForces(Real vDeltaT)
{
	if (m_BufferFlag)
	{
		m_ExternalForcesSolver->applyExternalForces(m_VelocityField1, vDeltaT);
	}
	else
	{
		m_ExternalForcesSolver->applyExternalForces(m_VelocityField2, vDeltaT);
	}
}

void CGridFluidSolver::_computeViscosity(Real vDeltaT)
{

}

void CGridFluidSolver::_computePressure(Real vDeltaT)
{
	if (m_BufferFlag)
	{
		if (typeid(*m_AdvectionSolver) == typeid(CParticleInCell))
		{
			m_PressureSolver->solvePressure
			(
				m_VelocityField1, 
				vDeltaT, 
				m_Boundarys.getTotalBoundarysVel(), 
				m_Boundarys.getTotalBoundarysSDF(),
				static_pointer_cast<CParticleInCell>(m_AdvectionSolver)->getEulerParticles().getParticlesPos()
			);
		}
		else if (typeid(*m_AdvectionSolver) == typeid(CFluidImplicitParticle))
		{
			m_PressureSolver->solvePressure
			(
				m_VelocityField1,
				vDeltaT,
				m_Boundarys.getTotalBoundarysVel(),
				m_Boundarys.getTotalBoundarysSDF(),
				static_pointer_cast<CFluidImplicitParticle>(m_AdvectionSolver)->getEulerParticles().getParticlesPos()
			);
		}
		else if (typeid(*m_AdvectionSolver) == typeid(CMixPICAndFLIP))
		{
			m_PressureSolver->solvePressure
			(
				m_VelocityField1,
				vDeltaT,
				m_Boundarys.getTotalBoundarysVel(),
				m_Boundarys.getTotalBoundarysSDF(),
				static_pointer_cast<CMixPICAndFLIP>(m_AdvectionSolver)->getEulerParticles().getParticlesPos()
			);
		}
		else if (typeid(*m_AdvectionSolver) == typeid(CSemiLagrangian))
		{
			m_PressureSolver->solvePressure(m_VelocityField1, vDeltaT, m_Boundarys.getTotalBoundarysVel(), m_Boundarys.getTotalBoundarysSDF(), m_FluidDomainField1);
		}
		else
		{

		}
	}
	else
	{
		if (typeid(*m_AdvectionSolver) == typeid(CParticleInCell))
		{
			m_PressureSolver->solvePressure
			(
				m_VelocityField2,
				vDeltaT,
				m_Boundarys.getTotalBoundarysVel(),
				m_Boundarys.getTotalBoundarysSDF(),
				static_pointer_cast<CParticleInCell>(m_AdvectionSolver)->getEulerParticles().getParticlesPos()
			);
		}
		else if (typeid(*m_AdvectionSolver) == typeid(CFluidImplicitParticle))
		{
			m_PressureSolver->solvePressure
			(
				m_VelocityField2,
				vDeltaT,
				m_Boundarys.getTotalBoundarysVel(),
				m_Boundarys.getTotalBoundarysSDF(),
				static_pointer_cast<CFluidImplicitParticle>(m_AdvectionSolver)->getEulerParticles().getParticlesPos()
			);
		}
		else if (typeid(*m_AdvectionSolver) == typeid(CMixPICAndFLIP))
		{
			m_PressureSolver->solvePressure
			(
				m_VelocityField2,
				vDeltaT,
				m_Boundarys.getTotalBoundarysVel(),
				m_Boundarys.getTotalBoundarysSDF(),
				static_pointer_cast<CMixPICAndFLIP>(m_AdvectionSolver)->getEulerParticles().getParticlesPos()
			);
		}
		else if (typeid(*m_AdvectionSolver) == typeid(CSemiLagrangian))
		{
			m_PressureSolver->solvePressure(m_VelocityField2, vDeltaT, m_Boundarys.getTotalBoundarysVel(), m_Boundarys.getTotalBoundarysSDF(), m_FluidDomainField2);
		}
		else
		{

		}
	}
}

void CGridFluidSolver::_extrapolatingVel()
{
	if (m_BufferFlag)
	{
		extrapolatingDataInvoker(m_VelocityField1, m_ExtrapolatingVelMarkersField, m_ExtrapolatingNums);
	}
	else
	{
		extrapolatingDataInvoker(m_VelocityField2, m_ExtrapolatingVelMarkersField, m_ExtrapolatingNums);
	}
}

void CGridFluidSolver::_fixFluidDomain()
{
	if (m_BufferFlag)
	{
		fixFluidDomainInvoker(m_Boundarys.getTotalBoundarysSDF(), m_FluidDomainField1);
	}
	else
	{
		fixFluidDomainInvoker(m_Boundarys.getTotalBoundarysSDF(), m_FluidDomainField2);
	}
}

void CGridFluidSolver::generateFluidDomainFromBBox(Vector3 vMin, Vector3 vMax)
{
	generateFluidDomainFromBBoxInvoker(vMin, vMax, m_FluidDomainField1);
	m_FluidDomainField2 = m_FluidDomainField1;
}

void CGridFluidSolver::generateSolidSDFFromCLDGrid(const std::shared_ptr<CCubicLagrangeDiscreteGrid>& vCLDGrid, Vector3 vVel)
{
	CCellCenteredScalarField TempSolidSDF(m_Resolution, m_Origin, m_Spacing);
	samplingSDFFromCLDGridInvoker(vCLDGrid, TempSolidSDF);
	m_Boundarys.addBoundary(TempSolidSDF, vVel);
}

void CGridFluidSolver::generateSolidSDFFromOBJFile(string vTriangleMeshFilePath, bool vIsInvSign)
{
	CCellCenteredScalarField TempSolidSDF(m_Resolution, m_Origin, m_Spacing);
	generateSDF(vTriangleMeshFilePath, TempSolidSDF, vIsInvSign);
	m_Boundarys.addBoundary(TempSolidSDF);
}

void CGridFluidSolver::generateCurFluidSDF()
{
	if (m_BufferFlag)
	{
		buildFluidSDFInvoker
		(
			m_FluidDomainField2,
			m_FluidSDFField,
			m_ExtrapolatingNums
		);
	}
	else
	{
		buildFluidSDFInvoker
		(
			m_FluidDomainField1,
			m_FluidSDFField,
			m_ExtrapolatingNums
		);
	}
}

void CGridFluidSolver::generateCurFluidDensity()
{
	if (m_BufferFlag)
	{
		buildFluidDensityInvoker
		(
			m_FluidDomainField2,
			m_Boundarys.getTotalBoundarysSDF(),
			m_FluidDensityField
		);
	}
	else
	{
		buildFluidDensityInvoker
		(
			m_FluidDomainField1,
			m_Boundarys.getTotalBoundarysSDF(),
			m_FluidDensityField
		);
	}
}

void CGridFluidSolver::mixVelFieldWithOthers
(
	const CFaceCenteredVectorField& vOtherVelField,
	const CCellCenteredScalarField& vOthersDensityField,
	const CCellCenteredScalarField& vFluidDensityField
)
{
	if (m_BufferFlag)
	{
		mixFieldWithDensityInvoker
		(
			vOtherVelField,
			m_VelocityField1,
			vOthersDensityField,
			vFluidDensityField
		);
	}
	else
	{
		mixFieldWithDensityInvoker
		(
			vOtherVelField,
			m_VelocityField2,
			vOthersDensityField,
			vFluidDensityField
		);
	}
}

void CGridFluidSolver::transferFluidVelField2Particles
(
	const thrust::device_vector<Real>& vParticlesPos,
	thrust::device_vector<Real>& voParticlesVel,
	EPGTransferAlgorithm vTransferAlg
)
{
	if (m_BufferFlag)
	{
		tranferFCVField2ParticlesInvoker
		(
			vParticlesPos,
			voParticlesVel,
			m_VelocityField2,
			vTransferAlg
		);
	}
	else
	{
		tranferFCVField2ParticlesInvoker
		(
			vParticlesPos,
			voParticlesVel,
			m_VelocityField1,
			vTransferAlg
		);
	}
}