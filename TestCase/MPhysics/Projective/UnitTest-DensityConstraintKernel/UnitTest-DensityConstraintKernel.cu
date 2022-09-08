#pragma once
#include "pch.h"

#include "CudaContextManager.h"
#include "BoundaryVolumeMap.h"
#include "ThrustWapper.cuh"
#include "DensityConstraintKernel.cuh"
#include "SimulationConfigManager.h"
#include "Particle.h"
#include "KNNSearch.h"

TEST(DensityConstraintKernel, solveDensityConstraints)
{
	//CCudaContextManager::getInstance().initCudaContext();
	//initAgzUtilsDX11Device();

	//SAABB Space;
	//Space.Min = Vector3(-0.15, -1.04, -0.15);
	//Space.Max = Vector3(0.15, -0.94, 0.15);
	//shared_ptr<CParticleGroup> RegularParticleData = make_shared<CParticleGroup>();
	//RegularParticleData->setParticleRadius(0.025);
	//RegularParticleData->appendParticleBlock(Space);
	//UInt ResultParticleSize = RegularParticleData->getSize();
	//UInt ConstraintsSize = ResultParticleSize;
	//cout << "Before Sort:" << endl;
	//cout << "Pos:" << endl << RegularParticleData->getParticlePos() << endl;
	//cout << "Vel:" << endl << RegularParticleData->getParticleVel() << endl;
	//cout << "PrevPos:" << endl << RegularParticleData->getPrevParticlePos() << endl;

	////加速度
	//Real TimeStep = CSimulationConfigManager::getInstance().getTimeStep();
	//CCuDenseVector Acceleration(ResultParticleSize * 3);
	//Acceleration.setZero();
	//computeAccelerationInvoker(Acceleration, CSimulationConfigManager::getInstance().getG());
	//RegularParticleData->addAccelerationToVel(Acceleration, TimeStep);

	////惯性项
	//CCuDenseVector S(ResultParticleSize * 3);
	//Real DeltaT = CSimulationConfigManager::getInstance().getTimeStep();
	//RegularParticleData->setPrevParticlePos(RegularParticleData->getConstParticlePos());
	//computeInertiaTermInvoker
	//(
	//	RegularParticleData->getParticlePosGPUPtr(),
	//	RegularParticleData->getConstParticleVelGPUPtr(),
	//	Acceleration.getConstVectorValueGPUPtr(),
	//	RegularParticleData->getSize(),
	//	DeltaT
	//);

	////Neighbor Search
	//CKNNSearch NeighborSearch;
	//NeighborSearch.bindParticleGroup(RegularParticleData);
	//NeighborSearch.search();

	//cout << "After Sort:" << endl;
	//cout << "Pos:" << endl << RegularParticleData->getParticlePos() << endl;
	//cout << "Vel:" << endl << RegularParticleData->getParticleVel() << endl;
	//cout << "PrevPos:" << endl << RegularParticleData->getPrevParticlePos() << endl;

	////Volume数据
	//shared_ptr<CRigidBodyBoundaryVolumeMap> Box = make_shared<CRigidBodyBoundaryVolumeMap>("./BoundaryCache_cube/");
	//Box->doInfluenceToParticle(*RegularParticleData);
	//std::vector<shared_ptr<CRigidBodyBoundaryVolumeMap>> Boxes;
	//Boxes.push_back(Box);

	////约束求解
	//thrust::device_vector<Vector3> P(NeighborSearch.getNeighorDataSize() + ConstraintsSize);
	//thrust::device_vector<Vector3> GradC(NeighborSearch.getNeighorDataSize() + ConstraintsSize);
	//CCuDenseVector Vectorb(ResultParticleSize * 3);
	//Vectorb.setZero();
	//initConstraintsInvoker
	//(
	//	RegularParticleData->getConstParticlePosGPUPtr(),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighorData()),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighborCounts()),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighorOffsets()),
	//	ConstraintsSize,
	//	getRawDevicePointerVector3(P)
	//);
	//solveDensityConstraintsInvoker
	//(
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighorData()),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighborCounts()),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighorOffsets()),
	//	Boxes,
	//	ConstraintsSize,
	//	RegularParticleData->getParticleVolume(),
	//	100,
	//	1e-14,
	//	RegularParticleData->getParticleSupportRadius(),
	//	getRawDevicePointerVector3(P),
	//	getRawDevicePointerVector3(GradC)
	//);
	//updateVectorbInvoker
	//(
	//	getReadOnlyRawDevicePointer(P),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighorData()),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighborCounts()),
	//	getReadOnlyRawDevicePointer(NeighborSearch.getNeighorOffsets()),
	//	50000,
	//	RegularParticleData->getSize(),
	//	getRawDevicePointerReal(Vectorb.getVectorValue())
	//);
	//Vectorb *= pow(CSimulationConfigManager::getInstance().getTimeStep(), 2);
	//Real Mass = RegularParticleData->getParticleVolume() * 1000;
	//Vectorb.plusAlphaX(RegularParticleData->getParticlePos(), Mass);

	//UInt CenterParticleIndex = 14;
	//UInt NeighborCount = NeighborSearch.getNeighborCount(CenterParticleIndex);
	//thrust::device_vector<Real> NeighorData(NeighborSearch.getNeighorData().size());
	//NeighorData = NeighborSearch.getNeighorData();
	//Vector3 ParticlePos = RegularParticleData->getParticlePos(CenterParticleIndex);
	//Vector3 b = ParticlePos(Vectorb, CenterParticleIndex);
	//Vector3 ClosestPoint = ParticlePos(Box->getBoundaryClosestPosCache(), CenterParticleIndex);
	//std::cout << "Particle :" << CenterParticleIndex << endl;
	//std::cout << "Particle Pos:" << ParticlePos.x << "," << ParticlePos.y << "," << ParticlePos.z << endl;
	//std::cout << "Boundary Volume:" << Box->getBoundaryVolumeCache()[CenterParticleIndex] << endl;
	//std::cout << "Boundary Dist:" << Box->getBoundaryDistCache()[CenterParticleIndex] << endl;
	//std::cout << "Closest Point:" << ClosestPoint.x << "," << ClosestPoint.y << "," << ClosestPoint.z << endl;
	//std::cout << "Vectorb Data:" << b.x << ", " << b.y << ", " << b.z << endl;
	//std::cout << "Neighbor Data:" << endl;
	//UInt NeighborOffset = NeighborSearch.getNeighorOffsets()[CenterParticleIndex];
	//for (UInt i = 0; i < NeighborCount; i++)
	//{
	//	UInt NeighborIndex = NeighorData[NeighborOffset + i];
	//	std::cout << "\t" << NeighborIndex <<endl;
	//}
	//for (UInt i = 0; i < NeighborCount + 1; i++)
	//{
	//	Vector3 PE = P[NeighborOffset + CenterParticleIndex + i];
	//	std::cout << "\t" << PE.x << "," << PE.y << "," << PE.z << endl;
	//}

	//std::cout << endl;

	//freeAgzUtilsDX11Device();
	//CCudaContextManager::getInstance().freeCudaContext();
}