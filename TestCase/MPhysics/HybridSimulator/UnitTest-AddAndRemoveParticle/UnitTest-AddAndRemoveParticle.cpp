#pragma once
#include "pch.h"
#include "Particle.h"
#include "KNNSearch.h"
#include "HybridSimulatorKernel.cuh"
#include "CellCenteredScalarField.h"
#include "CudaContextManager.h"

TEST(AddAndRemoveParticle, AddParticleInFluidSurface)
{
	CCudaContextManager::getInstance().initCudaContext();

	shared_ptr<CParticleGroup> Target = make_shared<CParticleGroup>();
	Target->setParticleRadius(0.025);
	Target->appendParticleBlock(SAABB(Vector3(0.25, 0.25, 0.25), Vector3(0.5, 0.5, 0.5)));

	CKNNSearch NeighborSearcher;
	NeighborSearcher.bindParticleGroup(Target);
	NeighborSearcher.search();

	CCellCenteredScalarField FluidDistC;
	CCellCenteredScalarField FluidRhoC;
	CCellCenteredScalarField FluidRhoG;

	Vector3i Res = Vector3i(4, 4, 4);
	UInt GridDataSize = Res.x * Res.y * Res.z;
	vector<Real> GridData(GridDataSize);
	Vector3 GridMin = Vector3(0, 0, 0);
	Vector3 GridMax = Vector3(1, 1, 1);
	Real CellSpace = 0.25;

	UInt SampleK = 20;
	Real vCreateDistRate = 2;
	Real vRhoMin = 0.55;
	Real vFluidRestDensity = 1000;

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (i == 1)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.5;
				}
				if (i == 0)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 1.0;
				}
			}
		}
	}
	FluidRhoG.resize(Res, GridMin, GridMax, GridData.data());
	FluidRhoC.resize(Res, GridMin, GridMax, GridData.data());

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (i == 1)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.125;
				}
				else if (i == 2)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.25;
				}
				else if (i == 3)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.5;
				}
				else
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.0;
				}
			}
		}
	}
	FluidDistC.resize(Res, GridMin, GridMax, GridData.data());

	UInt OldSize = Target->getSize();
	thrust::device_vector<UInt> NeedCache;
	resizeDeviceVector(NeedCache, Res.x * Res.y * Res.z);
	genParticleByPoissonDiskInvoker
	(
		FluidDistC.getConstGridDataGPUPtr(),
		FluidRhoC.getConstGridDataGPUPtr(),
		FluidRhoG.getGridDataGPUPtr(),

		Res,
		GridMin,
		GridDataSize,
		CellSpace,

		getReadOnlyRawDevicePointer(NeighborSearcher.getCellParticleOffsets()),
		getReadOnlyRawDevicePointer(NeighborSearcher.getCellParticleCounts()),
		NeighborSearcher.getGridInfo(),

		SampleK,
		vCreateDistRate,
		vRhoMin,
		vFluidRestDensity,
		Target,
		NeedCache
	);

	for (UInt i = OldSize; i < Target->getSize(); i++)
	{
		Vector3 Pos = Target->getParticlePos(i);

		ASSERT_GE(Pos.x, 0.0);
		ASSERT_LE(Pos.x, 1.0);

		ASSERT_GE(Pos.y, 0.0);
		ASSERT_LE(Pos.y, 1.0);

		ASSERT_GE(Pos.z, 0.25);
		ASSERT_LE(Pos.z, 0.5);

		Vector3 PrevPos = Target->getPrevParticlePos(i);
		ASSERT_EQ(Pos.x, PrevPos.x);
		ASSERT_EQ(Pos.y, PrevPos.y);
		ASSERT_EQ(Pos.z, PrevPos.z);

		Vector3 Vel = Target->getParticleVel(i);
		ASSERT_EQ(Vel.x, 0);
		ASSERT_EQ(Vel.y, 0);
		ASSERT_EQ(Vel.z, 0);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(AddAndRemoveParticle, RemoveParticleUnderWater)
{
	CCudaContextManager::getInstance().initCudaContext();

	shared_ptr<CParticleGroup> Target = make_shared<CParticleGroup>();
	Target->setParticleRadius(0.025);
	Target->appendParticleBlock(SAABB(Vector3(0.0, 0.0, 0.0), Vector3(0.2, 0.2, 1.0)));

	CKNNSearch NeighborSearcher;
	NeighborSearcher.bindParticleGroup(Target);
	NeighborSearcher.search();

	CCellCenteredScalarField FluidDistG;
	CCellCenteredScalarField FluidRhoC;
	CCellCenteredScalarField FluidRhoG;

	Vector3i Res = Vector3i(4, 4, 4);
	UInt GridDataSize = Res.x * Res.y * Res.z;
	vector<Real> GridData(GridDataSize);
	Vector3 GridMin = Vector3(0, 0, 0);
	Vector3 GridMax = Vector3(1, 1, 1);
	Real CellSpace = 0.25;

	UInt SampleK = 20;
	Real vCreateDistRate = 2;
	Real vRhoMin = 0.55;
	Real vFluidRestDensity = 1000;

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (i == 1)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.5;
				}
				if (i == 0)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 1.0;
				}
			}
		}
	}
	FluidRhoG.resize(Res, GridMin, GridMax, GridData.data());
	FluidRhoC.resize(Res, GridMin, GridMax, GridData.data());

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (i == 0)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.25;
				}
				else if (i == 1)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.125;
				}
				else if (i == 2)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.0;
				}
				else if (i == 3)
				{
					GridData[i * Res.x * Res.y + j * Res.x + k] = 0.0;
				}
			}
		}
	}
	FluidDistG.resize(Res, GridMin, GridMax, GridData.data());

	thrust::device_vector<bool> ShouldDelete;
	thrust::device_vector<Real> DistPG;
	resizeDeviceVector(DistPG, Target->getSize());
	resizeDeviceVector(ShouldDelete, Target->getSize() * 3, false);
	FluidDistG.sampleField(Target->getConstParticlePos().getConstVectorValue(), DistPG);

	deleteParticleUnderWater
	(
		getReadOnlyRawDevicePointer(DistPG),
		FluidRhoC.getConstGridDataGPUPtr(),
		FluidRhoG.getGridDataGPUPtr(),

		Res,
		GridMin,
		GridDataSize,
		CellSpace,
		Target->getSize(),
		CellSpace,
		3.0,
		0.01,
		1.0,
		ShouldDelete,
		Target
	);

	for (UInt i = 0; i < Target->getSize(); i++)
	{
		Vector3 Pos = Target->getParticlePos(i);
		std::cout << Pos.x << "," << Pos.y << "," << Pos.z << endl;
	}

	CCudaContextManager::getInstance().freeCudaContext();
}