#include "DensityConstraintKernel.cuh"

#include "Common.h"
#include "CudaContextManager.h"
#include "SPHKernelFunc.cuh"
#include "Particle.h"
#include "CuDenseVector.h"

__device__ Real computeCValue
(
	const Vector3* vP,
	const Vector3& vCenterPos,
	const UInt& vConstraintIndex,
	const UInt& vConstraintDataOffset,
	const UInt& vNeighborCount,
	const Real** vBoundaryVolume,
	const Real** vBoundaryClosestPoint,
	const UInt& vBoiundaryCount,
	const Real& vParticleVolume
)
{
	Real C = vParticleVolume * SmoothKernelCubic.W_zero();
	for (UInt i = 0; i < vNeighborCount; i++)
	{
		Vector3 NeighborPos = vP[i];
		C += vParticleVolume * SmoothKernelCubic.W(vCenterPos - NeighborPos);//rho / rho_0 = sum( m_j / W_ij ) / rho_0 = sum( V_j / W_ij )
	}
	
	for (UInt i = 0; i < vBoiundaryCount; i++)
	{
		Real BoundaryVolume = vBoundaryVolume[i][vConstraintIndex];
		Vector3 BoundaryClosestPoint = ParticlePos(vBoundaryClosestPoint[i], vConstraintIndex);
		if (BoundaryVolume > 0.0)
		{
			C += BoundaryVolume * SmoothKernelCubic.W(vCenterPos - BoundaryClosestPoint);
		}
	}

	C -= 1;
	if (C < 0) C = 0;

	return C;
}

__device__ Real computeGradC
(
	const Vector3* vP,
	const Vector3& vCenter,
	const UInt& vConstraintIndex,
	const UInt& vNeighborCount,
	const Real** vBoundaryVolume,
	const Real** vBoundaryClosestPoint,
	const UInt& vBoiundaryCount,
	const Real& vParticleVolume,
	Vector3& voCenterGradC
)
{
	Real GradCNorm2 = 0.0;
	voCenterGradC = Vector3(0, 0, 0);
	for (UInt i = 0; i < vNeighborCount; i++)
	{
		Vector3 NeighborPos = vP[i];
		Vector3 NeighborGrad = -vParticleVolume * SmoothKernelCubic.gradW(vCenter - NeighborPos);//GradCInConstraints = sum( m_j * gradW_ij) / rho_0
		GradCNorm2 += dot(NeighborGrad, NeighborGrad);
		voCenterGradC -= NeighborGrad;
	}

	for (UInt i = 0; i < vBoiundaryCount; i++)
	{
		Real BoundaryVolume = vBoundaryVolume[i][vConstraintIndex];
		Vector3 BoundaryClosestPoint = ParticlePos(vBoundaryClosestPoint[i], vConstraintIndex);
		if (BoundaryVolume > 0.0)
		{
			voCenterGradC += BoundaryVolume * SmoothKernelCubic.gradW(vCenter - BoundaryClosestPoint);
		}
	}

	GradCNorm2 += dot(voCenterGradC, voCenterGradC);

	return GradCNorm2;
}

__global__ void projectDensityConstraints
(
	const Real* vPos,
	const UInt* vNeighborData,
	const UInt* vNeighborCount,
	const UInt* vNeighborOffset,
	const Real** vBoundaryVolume,
	const Real** vBoundaryClosestPoint,
	UInt vBoundaryCount,
	UInt vConstraintSize,
	Real vParticleVolume,
	Real vStiffness,
	Real vDeltaTime,
	UInt vMaxIterationNum,
	Real vThreshold,

	Real* voVectorb
)
{
	UInt ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ThreadIndex >= vConstraintSize) return;

	UInt NeighborOffset = vNeighborOffset[ThreadIndex];
	UInt ConstraintOffset = NeighborOffset + ThreadIndex;
	UInt NeighborCount = vNeighborCount[ThreadIndex];

	UInt ItNum = 0;
	Real GradCNorm2 = REAL_MAX;
	Real C = REAL_MAX;

	Vector3 CenterPos = ParticlePos(vPos, ThreadIndex);
	Vector3 CenterGrad = Vector3(0, 0, 0);
	Vector3 PosInConstraints[MAXNeighborCount];

	for (UInt i = 0; i < NeighborCount; i++)
	{
		UInt NeighborIndex = vNeighborData[NeighborOffset + i];
		PosInConstraints[i] = ParticlePos(vPos, NeighborIndex);
	}

	C = computeCValue
	(
		PosInConstraints,
		CenterPos,
		ThreadIndex,
		ConstraintOffset,
		NeighborCount,
		vBoundaryVolume,
		vBoundaryClosestPoint,
		vBoundaryCount,
		vParticleVolume
	);

	while (ItNum < vMaxIterationNum && C > vThreshold)
	{
		//计算C和GradCInConstraints[N]
		GradCNorm2 = computeGradC
		(
			PosInConstraints,
			CenterPos,
			ThreadIndex,
			NeighborCount,
			vBoundaryVolume,
			vBoundaryClosestPoint,
			vBoundaryCount, 
			vParticleVolume, 
			CenterGrad
		);
		if (abs(GradCNorm2 - 0) < EPSILON) break;

		//使用得到的C和GradCInConstraints更新P
		Real Cdg = -C / (GradCNorm2 + 1e-6);
		for (UInt i = 0; i < NeighborCount; i++)
		{
			UInt NeighborIndex = vNeighborData[NeighborOffset + i];
			UInt NeighborCountOfCurrNeighbor = vNeighborCount[NeighborIndex];

			Vector3 NeighborGrad = -vParticleVolume * SmoothKernelCubic.gradW(CenterPos - PosInConstraints[i]);//GradCInConstraints = sum( m_j * gradW_ij) / rho_0
			PosInConstraints[i] += Cdg * static_cast<Real>(NeighborCountOfCurrNeighbor + 1) * NeighborGrad;
		}
		CenterPos += Cdg * static_cast<Real>(NeighborCount + 1) * CenterGrad;

		ItNum++;

		if (ItNum < vMaxIterationNum)
		{
			C = computeCValue
			(
				PosInConstraints,
				CenterPos,
				ThreadIndex,
				ConstraintOffset,
				NeighborCount,
				vBoundaryVolume,
				vBoundaryClosestPoint,
				vBoundaryCount,
				vParticleVolume
			);
		}
	}

	CenterPos *= (vStiffness * vDeltaTime * vDeltaTime);
	atomicAdd(&(ParticlePosPortion(voVectorb, ThreadIndex, 0)), CenterPos.x);
	atomicAdd(&(ParticlePosPortion(voVectorb, ThreadIndex, 1)), CenterPos.y);
	atomicAdd(&(ParticlePosPortion(voVectorb, ThreadIndex, 2)), CenterPos.z);
	for (UInt i = 0; i < NeighborCount; i++)
	{
		Vector3 NewP = (vStiffness * vDeltaTime * vDeltaTime) * PosInConstraints[i];
		UInt NeighborIndex = vNeighborData[NeighborOffset + i];
		atomicAdd(&(ParticlePosPortion(voVectorb, NeighborIndex, 0)), NewP.x);
		atomicAdd(&(ParticlePosPortion(voVectorb, NeighborIndex, 1)), NewP.y);
		atomicAdd(&(ParticlePosPortion(voVectorb, NeighborIndex, 2)), NewP.z);
	}
}

void initProjectiveSmoothKernelCubic(const CCubicKernel& SmoothKernelCubicCPU)
{
	CHECK_CUDA(cudaMemcpyToSymbol(SmoothKernelCubic, &SmoothKernelCubicCPU, sizeof(CCubicKernel)));
	CUDA_SYN
}

void solveDensityConstraintsInvoker
(
	const Real* vPos,
	const UInt* vNeighborData,
	const UInt* vNeighborCount,
	const UInt* vNeighborOffset,
	const vector<shared_ptr<CRigidBodyBoundaryVolumeMap>>& vBoundarys,
	UInt vConstraintSize,
	Real vParticleVolume,
	Real vStiffness,
	Real vDeltaTime,
	UInt vMaxIterationNum,
	Real vThreshold,
	CCuDenseVector& voVectorb
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vConstraintSize, BlockSize, GridSize, 0.75);

	thrust::device_vector<const Real*> Volumes;
	thrust::device_vector<const Real*> ClosestPoints;
	UInt BoundaryCount = 0;
	for (UInt i = 0; i < vBoundarys.size(); i++)
	{
		UInt InstanceCount = vBoundarys[i]->getInstanceCount();
		BoundaryCount += InstanceCount;
		for (UInt k = 0; k < InstanceCount; k++)
		{
			Volumes.push_back(raw_pointer_cast(vBoundarys[i]->getBoundaryVolumeCache(k).data()));
			ClosestPoints.push_back(raw_pointer_cast(vBoundarys[i]->getBoundaryClosestPosCache(k).data()));
		}
	}
	
	projectDensityConstraints LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vPos,
		vNeighborData,
		vNeighborCount,
		vNeighborOffset,
		raw_pointer_cast(Volumes.data()),
		raw_pointer_cast(ClosestPoints.data()),
		BoundaryCount,
		vConstraintSize,
		vParticleVolume,
		vStiffness,
		vDeltaTime,
		vMaxIterationNum,
		vThreshold,
		voVectorb.getVectorValueGPUPtr()
	);
	CUDA_SYN
} 