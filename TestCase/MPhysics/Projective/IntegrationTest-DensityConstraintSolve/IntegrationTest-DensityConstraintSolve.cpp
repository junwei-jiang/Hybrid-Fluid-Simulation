#pragma once
#include "pch.h"

#include "DensityConstraint.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"
#include "ThrustWapper.cuh"
#include "Particle.h"
#include "SimulationConfigManager.h"
#include "BoundaryVolumeMap.h"

class DensityConstraintsGroup : public testing::Test
{
protected:
	void SetUp() override
	{
		CCudaContextManager::getInstance().initCudaContext();
		initAgzUtilsDX11Device();

		SAABB Block;
		Block.Min = Vector3(-1.0, -0.95, -1.0);
		Block.Max = Vector3(0, 0, 0);

		RegularParticleData = make_shared<CParticleGroup>();
		RegularParticleData->generateParticleBlock(Block, 0.025);
		ResultParticleSize = RegularParticleData->getSize();

		RegularParticleData_backup = make_shared<CParticleGroup>();
		RegularParticleData_backup->generateParticleBlock(Block, 0.025);

		HighLightLog("PARTICLE NUM", to_string(ResultParticleSize));

		StaticBoundary = make_shared<CRigidBodyBoundaryVolumeMap>();
		StaticBoundary->bindBoundaryMesh(
			"./Cube.obj",
			Vector3ui(32, 32, 32),
			RegularParticleData->getParticleSupportRadius(),
			true
		);
	}

	void TearDown() override 
	{
		freeAgzUtilsDX11Device();
		CCudaContextManager::getInstance().freeCudaContext();
	}

protected:
	shared_ptr<CParticleGroup> RegularParticleData;
	shared_ptr<CParticleGroup> RegularParticleData_backup; 
	shared_ptr<CRigidBodyBoundaryVolumeMap> StaticBoundary;
	UInt ResultParticleSize;
};

TEST_F(DensityConstraintsGroup, WithGravity)
{
	CSimulationConfigManager::getInstance().setG(9.8);
	CSimulationConfigManager::getInstance().setTimeStep(0.016);

	CDensityConstraintGroup DensityConstraintGroup;
	DensityConstraintGroup.bindParticleGroup(RegularParticleData);
	Real ConstraintsSize = DensityConstraintGroup.getConstraintSize();

	SAABB SimulationDomain;
	SimulationDomain.Max = Vector3(1.0, 1.0, 1.0);
	SimulationDomain.Min = Vector3(-1.0, -1.0, -1.0);
	DensityConstraintGroup.setSimulationDomain(SimulationDomain);

	SGpuTimer GPUTimer;
	Real AvgTime = 0.0;
	UInt IterationNum = 1000;
	for (UInt i = 0; i < IterationNum; i++)
	{
		StaticBoundary->queryVolumeAndClosestPoint(*RegularParticleData);

		GPUTimer.Start();
		DensityConstraintGroup.solve();
		GPUTimer.Stop(); 
		for (UInt k = 0; k < ResultParticleSize; k++)
		{
			Vector3 Pos = RegularParticleData->getParticlePos(k);
			cout << "(" << Pos.x << "," << Pos.y << "," << Pos.z << ")" << endl;
		}
		cout << endl;
		if (i != 0) AvgTime += GPUTimer.Elapsed();
	}
	HighLightLog("AVG TIME", to_string(AvgTime / IterationNum));

	//for (UInt i = 0; i < ResultParticleSize; i++)
	//{
	//	Vector3 Pos = RegularParticleData->getParticlePos(i);
	//	Vector3 Pos_backup = RegularParticleData_backup->getParticlePos(i);

	//	EXPECT_LE(abs(Pos.x - Pos_backup.x), 1e-4);
	//	EXPECT_LE(abs(Pos.y - (Pos_backup.y - 13.7472)), 1e-4);
	//	EXPECT_LE(abs(Pos.z - Pos_backup.z), 1e-4);
	//}
}