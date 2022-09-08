#include "HybridSimulator.h"
#include "SimulationConfigManager.h"
#include "HybridSimulatorKernel.cuh"
#include "EulerParticlesTool.cuh"

CHybridSimulator::CHybridSimulator(const boost::property_tree::ptree& vSimualtorData)
{
	m_DistanceDamp = vSimualtorData.get_child("DistanceDamp").get_value<Real>();
	m_MixSubstepCount = vSimualtorData.get_child("MixSubstepCount").get_value<Real>();
	m_GRate = vSimualtorData.get_child("GRate").get_value<Real>();
	m_GFRate = vSimualtorData.get_child("GFRate").get_value<Real>();
	m_FRate = vSimualtorData.get_child("FRate").get_value<Real>();
	m_PRate = vSimualtorData.get_child("PRate").get_value<Real>();
	m_CreateRate = vSimualtorData.get_child("CreateRate").get_value<Real>();
	m_DeleteRate = vSimualtorData.get_child("DeleteRate").get_value<Real>();
	m_TimeDeleteRate = vSimualtorData.get_child("TimeDeleteRate").get_value<Real>();
	m_MaxParticleCount = vSimualtorData.get_child("MaxParticleCount").get_value<Real>();

	m_ProjectiveFluidSolver.loadConfg(vSimualtorData.get_child("ProjectiveSolver"));

	Real ParticleRadius = vSimualtorData.get_child("ParticleRadius").get_value<Real>();
	m_Particles = make_shared<CParticleGroup>();
	m_Particles->setParticleRadius(ParticleRadius);
	for (auto Boundary : vSimualtorData.get_child("Boundary"))
	{
		bool InvOutSide = Boundary.second.get_child("InvOutSide").get_value<bool>();
		bool IsDynamic = Boundary.second.get_child("IsDynamic").get_value<bool>();
		Vector3ui Resolution;
		Resolution.x = Boundary.second.get_child("Resolution.x").get_value<UInt>();
		Resolution.y = Boundary.second.get_child("Resolution.y").get_value<UInt>();
		Resolution.z = Boundary.second.get_child("Resolution.z").get_value<UInt>();

		vector<string> MeshPath;
		vector<SBoundaryTransform> MeshInitTransform;
		for (auto Mesh : Boundary.second.get_child("Mesh"))
		{
			MeshPath.push_back(Mesh.second.get_child("Path").get_value<string>());
			SBoundaryTransform Transform;
			Transform.Scale.x = Mesh.second.get_child("Scalar.x").get_value<Real>();
			Transform.Scale.y = Mesh.second.get_child("Scalar.y").get_value<Real>();
			Transform.Scale.z = Mesh.second.get_child("Scalar.z").get_value<Real>();
			Transform.Pos.x = Mesh.second.get_child("Pos.x").get_value<Real>();
			Transform.Pos.y = Mesh.second.get_child("Pos.y").get_value<Real>();
			Transform.Pos.z = Mesh.second.get_child("Pos.z").get_value<Real>();

			vector<Real> R;
			for (auto RNode : Mesh.second.get_child("Rotation"))
			{
				R.push_back(RNode.second.get_value<Real>());
			}
			Transform.Rotaion.row0 = Vector3(R[0], R[1], R[2]);
			Transform.Rotaion.row1 = Vector3(R[3], R[4], R[5]);
			Transform.Rotaion.row2 = Vector3(R[6], R[7], R[8]);
			MeshInitTransform.push_back(Transform);
		}
		__addBoundary(MeshPath, MeshInitTransform, Resolution, m_Particles->getParticleSupportRadius(), InvOutSide, IsDynamic);
	}

	//for (auto ParticleBlock : vSimualtorData.get_child("ParticleBlock"))
	//{
	//	SAABB ParticleBlockRange;
	//	ParticleBlockRange.Min.x = ParticleBlock.second.get_child("Range.Start.x").get_value<Real>();
	//	ParticleBlockRange.Min.y = ParticleBlock.second.get_child("Range.Start.y").get_value<Real>();
	//	ParticleBlockRange.Min.z = ParticleBlock.second.get_child("Range.Start.z").get_value<Real>();
	//	ParticleBlockRange.Max.x = ParticleBlock.second.get_child("Range.End.x").get_value<Real>();
	//	ParticleBlockRange.Max.y = ParticleBlock.second.get_child("Range.End.y").get_value<Real>();
	//	ParticleBlockRange.Max.z = ParticleBlock.second.get_child("Range.End.z").get_value<Real>();

	//	m_Particles->appendParticleBlock(ParticleBlockRange);
	//	__resizeParticlesCache();
	//}

	//Grid配置
	boost::property_tree::ptree GridSolverData = vSimualtorData.get_child("GridSolver");
	m_GridResolution.x = GridSolverData.get_child("GridResolution.x").get_value<Real>();
	m_GridResolution.y = GridSolverData.get_child("GridResolution.y").get_value<Real>();
	m_GridResolution.z = GridSolverData.get_child("GridResolution.z").get_value<Real>();

	Vector3 SimulationDomainStart;
	SimulationDomainStart.x = GridSolverData.get_child("SimulationDomain.Start.x").get_value<Real>();
	SimulationDomainStart.y = GridSolverData.get_child("SimulationDomain.Start.y").get_value<Real>();
	SimulationDomainStart.z = GridSolverData.get_child("SimulationDomain.Start.z").get_value<Real>();

	Vector3 SimulationDomainEnd;
	SimulationDomainEnd.x = GridSolverData.get_child("SimulationDomain.End.x").get_value<Real>();
	SimulationDomainEnd.y = GridSolverData.get_child("SimulationDomain.End.y").get_value<Real>();
	SimulationDomainEnd.z = GridSolverData.get_child("SimulationDomain.End.z").get_value<Real>();

	Vector3 SimulationDomainRange = SimulationDomainEnd - SimulationDomainStart;
	m_GridSpace = Vector3(SimulationDomainRange.x / m_GridResolution.x, SimulationDomainRange.y / m_GridResolution.y, SimulationDomainRange.z / m_GridResolution.z);

	m_GridMin = SimulationDomainStart - m_GridSpace;
	m_GridResolution += Vector3i(2, 2, 2);

	m_RhoGridG.resize(m_GridResolution, m_GridMin, m_GridSpace);
	m_RhoGridP.resize(m_GridResolution, m_GridMin, m_GridSpace);
	m_RhoGridC.resize(m_GridResolution, m_GridMin, m_GridSpace);

	m_ParticlesVelField.resize(m_GridResolution, m_GridMin, m_GridSpace);

	m_DistGridG.resize(m_GridResolution, m_GridMin, m_GridSpace);
	m_DistGridC.resize(m_GridResolution, m_GridMin, m_GridSpace);
	m_DistGridInside.resize(m_GridResolution, m_GridMin, m_GridSpace);

	m_CCSWeightField.resize(m_GridResolution, m_GridMin, m_GridSpace);
	m_FCVWeightField.resize(m_GridResolution, m_GridMin, m_GridSpace);

	m_DeltaGridVelField.resize(m_GridResolution, m_GridMin, m_GridSpace);

	resizeDeviceVector(m_NeedGenNewParticleCellIndexCache, m_GridResolution.x * m_GridResolution.y * m_GridResolution.z);

	m_GridFluidSolver.resizeGridFluidSolver(GridSolverData);
	m_GridFluidSolver.generateSolidSDFFromCLDGrid(m_ProjectiveFluidSolver.getBoundary()->getVolumeMap());
	m_GridFluidSolver.generateSolidSDFFromCLDGrid(m_ProjectiveFluidSolver.getBoundary(1)->getVolumeMap());
	m_GridFluidSolver.generateSolidSDFFromCLDGrid(m_ProjectiveFluidSolver.getBoundary(2)->getVolumeMap(), Vector3(-0.25, 0, 0));
	m_GridFluidSolver.generateFluid(GridSolverData);
}

void CHybridSimulator::update(Real vRealwroldDeltaTime)
{
	m_UpdateSteps++;

	m_GridFluidSolver.updateWithoutPressure();

	//__generateTempGridData();

	//__applyParticleInfluenceToGrid();

	m_GridFluidSolver.solvePressure();

	//m_ProjectiveFluidSolver.doNeighborSearch();

	//__keepParticleTypeSurface();

	//__applyGridInfluenceToParticle();
}

void CHybridSimulator::transformRigidBoundary
(
	UInt vStaticBoundaryIndex,
	UInt vInstanceIndex,
	SMatrix3x3 vRotation,
	Vector3 vPos
)
{
	m_RigidBoundary[vStaticBoundaryIndex]->transformInstance(vInstanceIndex, vRotation, vPos);
}

UInt CHybridSimulator::addRigidBoundaryInstance(UInt vStaticBoundaryIndex, SMatrix3x3 vRotation, Vector3 vPos)
{
	return m_RigidBoundary[vStaticBoundaryIndex]->addInstance(vRotation, vPos);
}

const std::shared_ptr<CParticleGroup>& CHybridSimulator::getTargetParticleGroup() const
{
	return m_Particles;
}

const CEulerParticles& CHybridSimulator::getEulerParticles() const
{
	if (typeid(*m_GridFluidSolver.getAdvectionSolver()) == typeid(CParticleInCell))
	{
		return static_pointer_cast<CParticleInCell>(m_GridFluidSolver.getAdvectionSolver())->getEulerParticles();
	}
	else if (typeid(*m_GridFluidSolver.getAdvectionSolver()) == typeid(CFluidImplicitParticle))
	{
		return static_pointer_cast<CFluidImplicitParticle>(m_GridFluidSolver.getAdvectionSolver())->getEulerParticles();
	}
	else if (typeid(*m_GridFluidSolver.getAdvectionSolver()) == typeid(CMixPICAndFLIP))
	{
		return static_pointer_cast<CMixPICAndFLIP>(m_GridFluidSolver.getAdvectionSolver())->getEulerParticles();
	}
}

const CCellCenteredScalarField& CHybridSimulator::getFluidSDF() const
{
	return m_DistGridG;
}

UInt CHybridSimulator::getConstNeighborDataSize() const
{
	return getDeviceVectorSize(m_ProjectiveFluidSolver.getNeighborSearch().getNeighorData());
}

const UInt * CHybridSimulator::getConstNeighborDataGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ProjectiveFluidSolver.getNeighborSearch().getNeighorData());
}

const UInt * CHybridSimulator::getConstNeighborCountGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ProjectiveFluidSolver.getNeighborSearch().getNeighborCounts());
}

const UInt* CHybridSimulator::getConstNeighborOffsetGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ProjectiveFluidSolver.getNeighborSearch().getNeighorOffsets());
}

Vector3 CHybridSimulator::getGridOrigin() const
{
	return m_GridMin;
}

Vector3 CHybridSimulator::getGridSpacing() const
{
	return m_GridSpace;
}

SGridInfo CHybridSimulator::getGridInfo() const
{
	return m_ProjectiveFluidSolver.getNeighborSearch().getGridInfo();
}

Real CHybridSimulator::getCellSize() const
{
	return m_ProjectiveFluidSolver.getNeighborSearch().getCellSize();
}

UInt CHybridSimulator::getCellDataSize() const
{
	return getDeviceVectorSize(m_ProjectiveFluidSolver.getNeighborSearch().getCellParticleCounts());
}

const UInt * CHybridSimulator::getConstCellParticleCountsGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ProjectiveFluidSolver.getNeighborSearch().getCellParticleCounts());
}

const UInt * CHybridSimulator::getConstCellParticleOffsetsGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ProjectiveFluidSolver.getNeighborSearch().getCellParticleOffsets());
}

void CHybridSimulator::__addBoundary
(
	const vector<std::string> & vTriangleMeshFilePath,
	const vector<SBoundaryTransform> & vBoundaryInitTransform,
	Vector3ui vRes,
	Real vExtension,
	bool isInv,
	bool vIsDynamic
)
{
	UInt Start = vTriangleMeshFilePath[0].find_last_of("/") + 1;
	UInt Size = vTriangleMeshFilePath[0].find_last_of(".") - Start;
	string Name = vTriangleMeshFilePath[0].substr(Start, Size);
	string Path = vTriangleMeshFilePath[0].substr(0, Start - 1);
	string CLDGridCacheFilePath = Path + "/StaticBoundaryCache_" + Name + "_"
		+ to_string(vRes.x) + "_" + to_string(vRes.y) + "_" + to_string(vRes.z) +
		+"_" + to_string(isInv) + "_" + to_string(vIsDynamic) +
		"/";

	shared_ptr<CRigidBodyBoundaryVolumeMap> RigidBodyBoundary;
	if (filesystem::exists(CLDGridCacheFilePath))
	{
		RigidBodyBoundary = make_shared<CRigidBodyBoundaryVolumeMap>(CLDGridCacheFilePath);
		m_RigidBoundary.push_back(RigidBodyBoundary);
	}
	else
	{
		filesystem::create_directory(CLDGridCacheFilePath);
		RigidBodyBoundary = make_shared<CRigidBodyBoundaryVolumeMap>();
		m_RigidBoundary.push_back(RigidBodyBoundary);
		m_RigidBoundary.back()->bindBoundaryMesh(
			vTriangleMeshFilePath,
			vBoundaryInitTransform,
			vRes,
			vExtension,
			isInv,
			false,
			CLDGridCacheFilePath.c_str()
		);
	}
	m_ProjectiveFluidSolver.addRigidBodyBoundary(RigidBodyBoundary);
}

void CHybridSimulator::__resizeParticlesCache()
{
	resizeDeviceVector(m_DistPGCache, m_Particles->getSize(), REAL_MAX);
	resizeDeviceVector(m_DistPG, m_Particles->getSize(), REAL_MAX);
	resizeDeviceVector(m_DistPC, m_Particles->getSize(), REAL_MAX);
	resizeDeviceVector(m_DistPInside, m_Particles->getSize(), REAL_MAX);
	resizeDeviceVector(m_Filter, m_Particles->getSize() * 3, false);
	resizeDeviceVector(m_PICVel, m_Particles->getSize() * 3, 0);
	resizeDeviceVector(m_FLIPVel, m_Particles->getSize() * 3, 0);
	resizeDeviceVector(m_FLIPVel, m_Particles->getSize() * 3, 0);
	resizeDeviceVector(m_DeltaParticlesVel, m_Particles->getSize() * 3, 0);
	resizeDeviceVector(m_ParticlesMass, m_Particles->getSize(), m_Particles->getParticleVolume());
	m_PredictPos.resize(m_Particles->getSize() * 3, 0);
	m_PredictVel.resize(m_Particles->getSize() * 3, 0);
	m_ProjectiveFluidSolver.updateTargetParticleGroup(m_Particles);
}

void CHybridSimulator::__keepParticleTypeSurface()
{
	if (m_Particles->getSize() <= m_MaxParticleCount)
	{
		//使用泊松圆盘分布在符合添加条件的格子内添加粒子
		genParticleByPoissonDiskInvoker
		(
			m_DistGridC.getConstGridDataGPUPtr(),
			m_RhoGridC.getConstGridDataGPUPtr(),
			m_RhoGridG.getGridDataGPUPtr(),
			m_GridResolution,
			m_GridMin,
			m_GridResolution.x * m_GridResolution.y * m_GridResolution.z,
			m_GridSpace.x,
			getReadOnlyRawDevicePointer(m_ProjectiveFluidSolver.getNeighborSearch().getNeighorOffsets()),
			getReadOnlyRawDevicePointer(m_ProjectiveFluidSolver.getNeighborSearch().getNeighborCounts()),
			m_ProjectiveFluidSolver.getNeighborSearch().getGridInfo(),
			m_SampleK,
			m_CreateRate,
			m_RhoMin,
			m_ProjectiveFluidSolver.getRestDensity(),
			m_Particles,
			m_NeedGenNewParticleCellIndexCache
		);
		__resizeParticlesCache();
	}

	//m_DistGridInside.sampleField(m_Particles->getConstParticlePos().getConstVectorValue(), m_DistPInside, ESamplingAlgorithm::TRILINEAR);
	//deleteParticleUnderWater
	//(
	//	getReadOnlyRawDevicePointer(m_DistPInside),
	//	m_Particles->getConstParticlePosGPUPtr(),
	//	m_RhoGridG.getGridDataGPUPtr(),
	//	m_GridResolution,
	//	m_GridMin,
	//	m_GridResolution.x * m_GridResolution.y * m_GridResolution.z,
	//	m_GridSpace.x,
	//	m_Particles->getSize(),
	//	m_GridSpace.x,
	//	m_DeleteRate,
	//	CSimulationConfigManager::getInstance().getTimeStep(),
	//	CSimulationConfigManager::getInstance().getTimeStep() * m_LiveRate,
	//	m_Filter,
	//	m_Particles
	//);
	//__resizeParticlesCache();
}

void CHybridSimulator::__applyParticleInfluenceToGrid()
{
	m_GridFluidSolver.mixVelFieldWithOthers(m_ParticlesVelField, m_RhoGridP, m_RhoGridG);
}

void CHybridSimulator::__applyGridInfluenceToParticle()
{
	//未混合的网格GradG计算到液面的距离，并插值到 m_DistPGCache
	m_DistGridG.sampleField(m_Particles->getConstParticlePos().getConstVectorValue(), m_DistPG, ESamplingAlgorithm::TRILINEAR);

	m_DeltaGridVelField.plusAlphaX(m_GridFluidSolver.getVelocityField(), -1.0);
	m_DeltaGridVelField *= -1.0;

	//smoothAccumlate(m_DistPGCache, m_DistanceDamp, m_DistPG);
	Real SubDeltaT = CSimulationConfigManager::getInstance().getTimeStep() / (Real)(m_MixSubstepCount);
	m_ProjectiveFluidSolver.setDeltaT(SubDeltaT);
	for (UInt i = 0; i < m_MixSubstepCount; i++)
	{
		__predict(SubDeltaT);

		//在网格上针对预测的位置与速度进行PIC和FLIP计算
		m_GridFluidSolver.transferFluidVelField2Particles(m_PredictPos.getConstVectorValue(), m_PICVel, EPGTransferAlgorithm::LINEAR);
		tranferFCVField2ParticlesInvoker
		(
			m_PredictPos.getConstVectorValue(),
			m_DeltaParticlesVel,
			m_DeltaGridVelField,
			EPGTransferAlgorithm::LINEAR
		);
		m_FLIPVel = m_PredictVel.getConstVectorValue();
		axpyReal(m_DeltaParticlesVel, m_FLIPVel, (Real)(1.0 / m_MixSubstepCount));

		mixParitcleVelWithPICAndFLIPInvoker
		(
			getReadOnlyRawDevicePointer(m_PICVel),
			getReadOnlyRawDevicePointer(m_FLIPVel),
			getReadOnlyRawDevicePointer(m_DistPG),
			m_Particles->getSize(),
			m_GridSpace.x,
			m_GRate,
			m_GFRate,
			m_FRate,
			m_PRate,
			m_PredictVel.getVectorValueGPUPtr()
		);
		m_Particles->setParticleVel(m_PredictVel);

		//SPH解算
		m_Particles->freshParticlePos(EXPLICIT_EULER, SubDeltaT);
		m_ProjectiveFluidSolver.doNeighborSearch();
		m_ProjectiveFluidSolver.projectAndSolveLiner();
		m_Particles->freshParticleVel(EXPLICIT_EULER, SubDeltaT);
		m_ProjectiveFluidSolver.applyBoundaryeffect();
	}
}

void CHybridSimulator::__generateTempGridData()
{
	__predict(CSimulationConfigManager::getInstance().getTimeStep());

	tranferParticles2CCSFieldInvoker
	(
		m_PredictPos.getConstVectorValue(),
		m_ParticlesMass,
		m_RhoGridP,
		m_CCSWeightField,
		EPGTransferAlgorithm::LINEAR,
		false
	);

	m_RhoGridP *= (1 / (m_GridSpace.x * m_GridSpace.y * m_GridSpace.z));

	tranferParticles2FCVFieldInvoker
	(
		m_PredictPos.getConstVectorValue(),
		m_PredictVel.getConstVectorValue(),
		m_ParticlesVelField,
		m_FCVWeightField,
		EPGTransferAlgorithm::LINEAR
	);

	buildFluidDensityInvoker
	(
		m_GridFluidSolver.getFluidDomainFieldBeforePressure(),
		m_GridFluidSolver.getSolidSDFField(),
		m_RhoGridG
	);

	m_RhoGridC = m_RhoGridG;
	m_RhoGridC += m_RhoGridP;

	buildFluidOutsideSDFInvoker
	(
		m_RhoGridG,
		m_GridFluidSolver.getSolidSDFField(),
		m_DistGridG,
		1000
	);

	buildFluidInsideSDFInvoker
	(
		m_RhoGridG,
		m_GridFluidSolver.getSolidSDFField(),
		m_DistGridInside,
		1000
	);

	buildMixedFluidOutsideSDFInvoker
	(
		m_RhoGridG,
		m_RhoGridC,
		m_DistGridG,
		m_DistGridC
	);

	m_DistGridG *= m_GridSpace.x;
	m_DistGridC *= m_GridSpace.x;

	m_DeltaGridVelField = m_GridFluidSolver.getVelocityFieldBeforePressure();
}

void CHybridSimulator::__predict(Real vDeltaTime)
{
	m_PredictVel = m_Particles->getConstParticleVel();

	predictParticleVelInvoker
	(
		m_PredictVel.getVectorValue(),
		m_Particles->getSize(),
		Vector3(0.0, -CSimulationConfigManager::getInstance().getG(), 0.0),
		vDeltaTime
	);

	m_PredictPos = m_PredictVel;
	m_PredictPos *= vDeltaTime;
	m_PredictPos += m_Particles->getConstParticlePos();
}
