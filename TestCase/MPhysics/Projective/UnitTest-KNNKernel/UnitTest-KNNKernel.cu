#pragma once
#include "pch.h"

#define NOMINMAX
#include <windows.h>

#include "KNNKernel.cuh"
#include "ThrustWapper.cuh"
#include "CudaContextManager.h"

__global__ void generate3DPosData(Real* voPosData, UInt vPosSize)
{
	int X = threadIdx.x;
	int Y = threadIdx.y;
	int Z = threadIdx.z;
	int Index =  16 * Z + 4 * Y + X;
	voPosData[Index * 3] = 0.125 + X * 0.25;
	voPosData[Index * 3 + 1] = 0.125 + Y * 0.25;
	voPosData[Index * 3 + 2] = 0.125 + Z * 0.25;
}

class KNNKernel : public testing::Test
{
public:
	void SetUp() override
	{
		CCudaContextManager::getInstance().initCudaContext();

		GridInfo.SearchRadius = 0.26;
		GridInfo.AABB.Min = Vector3(0, 0, 0);
		GridInfo.AABB.Max = Vector3(1, 1, 1);
		GridInfo.MetaGridGroupSize = 2;
		GridInfo.MetaGridBlockSize = 2 * 2 * 2;
		GridInfo.MetaGridDimension = Vector3ui(2, 2, 2);
		GridInfo.GridDimension = Vector3ui(4, 4, 4);
		GridInfo.GridDelta = Vector3(4, 4, 4);
		GridInfo.CellCount = 64;

		UInt X = GridInfo.GridDimension.x;
		UInt Z = GridInfo.GridDimension.z;
		UInt Y = GridInfo.GridDimension.y;
		ParticleSize = X * Y * Z;
		ParticlePosition.resize(ParticleSize * 3);
		generate3DPosData<<<1, dim3(4, 4, 4)>>>(raw_pointer_cast(ParticlePosition.data()), ParticleSize);
	}

	void TearDown() override
	{
		CCudaContextManager::getInstance().freeCudaContext();
	}

protected:
	UInt ParticleSize;
	thrust::device_vector<Real> ParticlePosition;
	SGridInfo GridInfo;
};

TEST_F(KNNKernel, computeCellInformation)
{
	thrust::device_vector<UInt> ParticleCellIndices(ParticleSize);
	thrust::device_vector<UInt> TempSortIndices(ParticleSize);
	thrust::device_vector<UInt> SortIndices(ParticleSize * 3);

	UInt CellCount = 64;
	thrust::device_vector<UInt> CellParticleCounts(CellCount);
	thrust::device_vector<UInt> CellOffsets(CellCount);

	computeCellInformation(
		raw_pointer_cast(ParticlePosition.data()),
		raw_pointer_cast(ParticleCellIndices.data()),
		raw_pointer_cast(TempSortIndices.data()),
		raw_pointer_cast(SortIndices.data()),
		ParticleSize,
		raw_pointer_cast(CellParticleCounts.data()),
		raw_pointer_cast(CellOffsets.data()),
		GridInfo
	);

	UInt RightResult[64] = { 0,1,4,5,16,17,20,21,2,3,6,7,18,19,22,23,8,9,12,13,24,25,28,29,10,11,14,15,26,27,30,31,32,33,36,37,48,49,52,53,34,35,38,39,50,51,54,55,40,41,44,45,56,57,60,61,42,43,46,47,58,59,62,63 };
	for (UInt i = 0; i < ParticleSize; i++)
	{
		EXPECT_EQ(SortIndices[i * 3], RightResult[i] * 3);
		EXPECT_EQ(SortIndices[i * 3 + 1], RightResult[i] * 3 + 1);
		EXPECT_EQ(SortIndices[i * 3 + 2], RightResult[i] * 3 + 2);
	}
}

#define SIZE 1000000
#define OFFSET 4
TEST_F(KNNKernel, computeMinMax)
{
	SAABB RandomResult;
	thrust::host_vector<Real> PosDataCPU;
	srand((unsigned)time(NULL));
	for (int i = 0; i < SIZE * 3; i++)
	{
		float a = rand();
		PosDataCPU.push_back(a);
	}
	thrust::device_vector<Real> PosDataRadom(SIZE * 3);
	PosDataRadom = PosDataCPU;

	_LARGE_INTEGER time_start;
	_LARGE_INTEGER time_end;
	double dqFreq;
	LARGE_INTEGER f;
	QueryPerformanceFrequency(&f);
	dqFreq = (double)f.QuadPart;
	QueryPerformanceCounter(&time_start);
	computeMinMax(thrust::raw_pointer_cast(PosDataRadom.data()), SIZE, 1, RandomResult);
	QueryPerformanceCounter(&time_end);
	HighLightLog("Random data cost time",
		to_string(1000 * (time_end.QuadPart - time_start.QuadPart) / dqFreq) + " ms");

	SAABB Result;
	thrust::device_ptr<Real> PosDataSequence = thrust::device_malloc<Real>(SIZE * 3);
	thrust::sequence(PosDataSequence, PosDataSequence + SIZE, OFFSET);
	thrust::sequence(PosDataSequence + SIZE, PosDataSequence + 2 * SIZE, OFFSET);
	thrust::sequence(PosDataSequence + 2 * SIZE, PosDataSequence + 3 * SIZE, OFFSET);

	QueryPerformanceCounter(&time_start);
	computeMinMax(raw_pointer_cast(PosDataSequence), SIZE, 1, Result);
	QueryPerformanceCounter(&time_end);

	HighLightLog("Sequence data cost time",
		to_string(1000 * (time_end.QuadPart - time_start.QuadPart) / dqFreq) + " ms");

	EXPECT_FLOAT_EQ(Result.Min.x, OFFSET);
	EXPECT_FLOAT_EQ(Result.Min.y, OFFSET);
	EXPECT_FLOAT_EQ(Result.Min.z, OFFSET);

	EXPECT_FLOAT_EQ(Result.Max.x, SIZE + OFFSET - 1);
	EXPECT_FLOAT_EQ(Result.Max.y, SIZE + OFFSET - 1);
	EXPECT_FLOAT_EQ(Result.Max.z, SIZE + OFFSET - 1);
}

TEST_F(KNNKernel, sortParticleData)
{
	thrust::device_vector<Real> ParticleVelicity(ParticleSize * 3);
	thrust::device_vector<Real> ParticlePrevPosition(ParticleSize * 3);
	thrust::device_vector<Real> ParticleLiveTime(ParticleSize * 3);
	thrust::sequence(ParticleVelicity.begin(), ParticleVelicity.end(), (Real)0.0, (Real)0.1);
	thrust::sequence(ParticlePrevPosition.begin(), ParticlePrevPosition.end(), (Real)0.0, (Real)0.1);
	thrust::sequence(ParticleLiveTime.begin(), ParticleLiveTime.end(), (Real)0.0, (Real)0.1);

	thrust::device_vector<UInt> ParticleCellIndices(ParticleSize);
	thrust::device_vector<UInt> TempSortIndices(ParticleSize);
	thrust::device_vector<UInt> SortIndices(ParticleSize * 3);

	UInt CellCount = 64;
	thrust::device_vector<UInt> CellParticleCounts(CellCount);
	thrust::device_vector<UInt> CellOffsets(CellCount);

	computeCellInformation(
		raw_pointer_cast(ParticlePosition.data()),
		raw_pointer_cast(ParticleCellIndices.data()),
		raw_pointer_cast(TempSortIndices.data()),
		raw_pointer_cast(SortIndices.data()),
		ParticleSize,
		raw_pointer_cast(CellParticleCounts.data()),
		raw_pointer_cast(CellOffsets.data()),
		GridInfo
	);

	thrust::device_vector<Real> ParticlePosition_Back(ParticleSize * 3);
	ParticlePosition_Back = ParticlePosition;
	thrust::device_vector<Real> ParticleVelicity_Back(ParticleSize * 3);
	ParticleVelicity_Back = ParticleVelicity;
	thrust::device_vector<Real> ParticlePrevPosition_Back(ParticleSize * 3);
	ParticlePrevPosition_Back = ParticlePrevPosition;
	thrust::device_vector<Real> SortCache(ParticleSize * 3);
	zsortParticleGroup
	(
		ParticlePosition,
		ParticleVelicity,
		ParticlePrevPosition,
		ParticleLiveTime,
		SortCache,
		SortIndices
	);

	UInt RightResult[64] = { 0,1,4,5,16,17,20,21,2,3,6,7,18,19,22,23,8,9,12,13,24,25,28,29,10,11,14,15,26,27,30,31,32,33,36,37,48,49,52,53,34,35,38,39,50,51,54,55,40,41,44,45,56,57,60,61,42,43,46,47,58,59,62,63 };
	for (UInt i = 0; i < ParticleSize; i++)
	{
		Vector3 ParticlePoss = ParticlePos(ParticlePosition, i);
		Vector3 ParticleVel = ParticlePos(ParticleVelicity, i);
		Vector3 ParticlePrevPos = ParticlePos(ParticlePrevPosition, i);
		Vector3 OldParticlePos = ParticlePos(ParticlePosition_Back, RightResult[i]);
		Vector3 OldParticleVel = ParticlePos(ParticleVelicity_Back, RightResult[i]);
		Vector3 OldParticlePrevPos = ParticlePos(ParticlePrevPosition_Back, RightResult[i]);
		ASSERT_LE(abs(OldParticlePos.x - ParticlePoss.x), EPSILON);
		ASSERT_LE(abs(OldParticlePos.y - ParticlePoss.y), EPSILON);
		ASSERT_LE(abs(OldParticlePos.z - ParticlePoss.z), EPSILON);

		ASSERT_LE(abs(OldParticleVel.x - ParticleVel.x), EPSILON);
		ASSERT_LE(abs(OldParticleVel.y - ParticleVel.y), EPSILON);
		ASSERT_LE(abs(OldParticleVel.z - ParticleVel.z), EPSILON);

		ASSERT_LE(abs(OldParticlePrevPos.x - ParticlePrevPos.x), EPSILON);
		ASSERT_LE(abs(OldParticlePrevPos.y - ParticlePrevPos.y), EPSILON);
		ASSERT_LE(abs(OldParticlePrevPos.z - ParticlePrevPos.z), EPSILON);
	}
}

class KNNKernel_Add : public KNNKernel
{
public:
	void SetUp() override
	{
		CCudaContextManager::getInstance().initCudaContext();

		GridInfo.SearchRadius = 0.708;
		GridInfo.AABB.Min = Vector3(0, 0, 0);
		GridInfo.AABB.Max = Vector3(1, 1, 1);
		GridInfo.MetaGridGroupSize = 2;
		GridInfo.MetaGridBlockSize = 2 * 2 * 2;
		GridInfo.MetaGridDimension = Vector3ui(2, 2, 2);
		GridInfo.GridDimension = Vector3ui(4, 4, 4);
		GridInfo.GridDelta = Vector3(4, 4, 4);
		GridInfo.CellCount = 64;

		UInt X = GridInfo.GridDimension.x;
		UInt Z = GridInfo.GridDimension.z;
		UInt Y = GridInfo.GridDimension.y;
		ParticleSize = X * Y * Z;
		ParticlePosition.resize(ParticleSize * 3);
		generate3DPosData << <1, dim3(4, 4, 4) >> > (raw_pointer_cast(ParticlePosition.data()), ParticleSize);

		ParticlePosition.resize(ParticleSize * 3 + 5 * 3);
		ParticlePosition[ParticleSize * 3 + 0 + 0] = 0.125 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 0 + 1] = 0.125 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 0 + 2] = 0.125;

		ParticlePosition[ParticleSize * 3 + 3 + 0] = 0.25 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 3 + 1] = 0.25 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 3 + 2] = 0.125;

		ParticlePosition[ParticleSize * 3 + 6 + 0] = 0.125 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 6 + 1] = 0.125 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 6 + 2] = 0.375;

		ParticlePosition[ParticleSize * 3 + 9 + 0] = 0.25 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 9 + 1] = 0.25 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 9 + 2] = 0.375;

		ParticlePosition[ParticleSize * 3 + 12 + 0] = 0.375 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 12 + 1] = 0.375 + 0.0625;
		ParticlePosition[ParticleSize * 3 + 12 + 2] = 0.375;
		ParticleSize += 5;
	}
};

TEST_F(KNNKernel_Add, searchNeighbor)
{
	thrust::device_vector<Real> ParticleVel(ParticleSize * 3);
	thrust::device_vector<Real> ParticlePrevPos(ParticleSize * 3);
	thrust::device_vector<Real> ParticleLiveTime(ParticleSize * 3);

	thrust::device_vector<UInt> ParticleCellIndices(ParticleSize);
	thrust::device_vector<UInt> TempSortIndices(ParticleSize);
	thrust::device_vector<UInt> SortIndices(ParticleSize * 3);

	UInt CellCount = 64;
	thrust::device_vector<UInt> CellParticleCounts(CellCount);
	thrust::device_vector<UInt> CellOffsets(CellCount);

	computeCellInformation(
		raw_pointer_cast(ParticlePosition.data()),
		raw_pointer_cast(ParticleCellIndices.data()),
		raw_pointer_cast(TempSortIndices.data()),
		raw_pointer_cast(SortIndices.data()),
		ParticleSize,
		raw_pointer_cast(CellParticleCounts.data()),
		raw_pointer_cast(CellOffsets.data()),
		GridInfo
	);
	thrust::device_vector<Real> ParticlePosition_Back(ParticleSize * 3);
	ParticlePosition_Back = ParticlePosition;


	thrust::device_vector<Real> SortCache(ParticleSize * 3);
	zsortParticleGroup
	(
		ParticlePosition,
		ParticleVel,
		ParticlePrevPos,
		ParticleLiveTime,
		SortCache,
		SortIndices
	);

	HighLightLog("IndexData", ":");
	outputUIntVector(CellOffsets);
	outputUIntVector(SortIndices);
	outputUIntVector(CellParticleCounts);

	thrust::device_vector<UInt> NeighborOffsets(ParticleSize);
	thrust::device_vector<UInt> NeighborCounts(ParticleSize);
	thrust::device_vector<UInt> Neighbors(ParticleSize);
	searchNeighbors
	(
		raw_pointer_cast(ParticlePosition.data()),
		ParticleSize,
		raw_pointer_cast(CellParticleCounts.data()),
		raw_pointer_cast(CellOffsets.data()),
		GridInfo,
		NeighborOffsets,
		NeighborCounts,
		Neighbors
	);

	for (int i = 0; i < ParticleSize; i++)
	{
		UInt NeighborCount = NeighborCounts[i];
		UInt NeighborOffset = NeighborOffsets[i];
		cout << "Particle " << i << ":" << endl;
		for (UInt i = NeighborOffset; i < NeighborOffset + NeighborCount; i++)
		{
			cout << "\t" << Neighbors[i] << endl;
		}
	}
}