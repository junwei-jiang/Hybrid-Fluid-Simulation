#pragma once
#include "pch.h"

#include "KNNSearch.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"
#include "ThrustWapper.cuh"
#include "Particle.h"

void doForceTest(shared_ptr<CParticleGroup> vParticleData, CKNNSearch& vNeighborSearcher, bool vIsProfile = false)
{
	Real TotalTime = 0;
	UInt IterationNum = 24;
	for (UInt i = 0; i < IterationNum; i++)
	{
		SGpuTimer Timer;
		Timer.Start();
		vNeighborSearcher.search();
		Timer.Stop();
		if (i != 0) TotalTime += Timer.Elapsed();
	}
	HighLightLog("AVGERAGE TIME", to_string(TotalTime / IterationNum) + " ms");
	cout << endl;

	if (!vIsProfile)
	{
		map<UInt, unordered_set<UInt>> NeighborGPUInfo;
		vNeighborSearcher.getNeighborsDebug(NeighborGPUInfo);

		for (auto Pair : NeighborGPUInfo)
		{
			UInt ParticleIndex = Pair.first;
			Vector3 CurrPos = vParticleData->getParticlePos(ParticleIndex);
			for (UInt Neighbor : Pair.second)
			{
				Vector3 NeighborPos = vParticleData->getParticlePos(Neighbor);
				Real Dis = length(CurrPos - NeighborPos);
				EXPECT_TRUE(Dis <= vNeighborSearcher.getSearchRadius())
					<< "Particle:" << ParticleIndex << " GPUSetSize:" << Pair.second.size();
			}
		}
	}
}

TEST(NeighborSearch, RandomDataTest)
{
	CCudaContextManager::getInstance().initCudaContext();

	SAABB Space;
	Space.Min = Vector3(-1, -1, -1);
	Space.Max = Vector3(1, 1, 1);
	Real ParticleRadius = 0.125;

	shared_ptr<CParticleGroup> RandomParticleData = make_shared<CParticleGroup>();
	RandomParticleData->generateRandomData(Space, ParticleRadius);

	CKNNSearch NeighborSearcher;
	NeighborSearcher.bindParticleGroup(RandomParticleData);

	doForceTest(RandomParticleData, NeighborSearcher);

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(NeighborSearch, Regular4x4GridTest)
{
	CCudaContextManager::getInstance().initCudaContext();

	SAABB Space;
	Space.Min = Vector3(-1, -1, -1);
	Space.Max = Vector3(1, 1, 1);

	shared_ptr<CParticleGroup> RegularParticleData = make_shared<CParticleGroup>();
	RegularParticleData->setParticleRadius(0.125);
	RegularParticleData->appendParticleBlock(Space);

	CKNNSearch NeighborSearcher;
	NeighborSearcher.bindParticleGroup(RegularParticleData);

	doForceTest(RegularParticleData, NeighborSearcher);

	CCudaContextManager::getInstance().freeCudaContext();
}