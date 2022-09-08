#include "BoundaryVolumeMapKernel.cuh"
#include "CudaContextManager.h"
#include "CLDGridinterpolate.cuh"
#include "GaussQuadrature.cuh"
#include "SPHKernelFunc.cuh"

__device__ Real __integrate
(
	const Vector3& vX,
	const Vector3& vXi,
	Real vSupportRadius,
	const Real* vSDFField,
	const UInt* vCellData,
	SCLDGridInfo vCLDGridInfo
)
{
	if (dot(vXi, vXi) > vSupportRadius * vSupportRadius)
		return 0.0;

	Real XiDist;
	interpolateSinglePos(vX + vXi, vSDFField, vCellData, vCLDGridInfo, XiDist);

	if (XiDist <= 0.0)
		return 1.0;// -0.001 * dist / supportRadius;
	if (XiDist < 1.0 / vSupportRadius)
		return static_cast<double>(SmoothKernelCubic.W(static_cast<Real>(XiDist)) / SmoothKernelCubic.W_zero());

	return 0.0;
}

//P = 30，16个采样点的gauss积分式
__device__  Real integrateGaussQuadrature
(
	const SAABB & vDomain,
	Real vSupportRadius,
	Vector3 vNodePos,
	const Real* vSDFField,
	const UInt* vCellData,
	SCLDGridInfo vCLDGridInfo
)
{
	Vector3 C0 = 0.5 * vDomain.getDia();
	Vector3 C1 = 0.5 * (vDomain.Min + vDomain.Max);

	Real Result = 0.0;
	Vector3 Xi = Vector3(0, 0, 0);
	for (UInt i = 0; i < 16; ++i)
	{
		Real Wi = gaussian_weights_1_30[i];
		Xi.x = gaussian_abscissae_1_30[i];
		for (UInt j = 0; j < 16; ++j)
		{
			Real Wij = Wi * gaussian_weights_1_30[j];
			Xi.y = gaussian_abscissae_1_30[j];
			for (UInt k = 0; k < 16; ++k)
			{
				Real Wijk = Wij * gaussian_weights_1_30[k];
				Xi.z = gaussian_abscissae_1_30[k];

				Result += Wijk * __integrate(vNodePos, C0 * Xi + C1, vSupportRadius, vSDFField, vCellData, vCLDGridInfo);
			}
		}
	}

	Result *= (C0.x * C0.y * C0.z);
	return Result;
}

__global__ void computeVolumeValue
(
	SCLDGridInfo vCLDGridInfo,
	Real* voGridNodeValue,
	const UInt* vCellData,
	const Real* vSDFField,
	Real vSupportRadius
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vCLDGridInfo.TotalNodeNum) return;
	Vector3 NodePos = vCLDGridInfo.indexToNodePosition(Index);

	Real Dist;
	interpolateSinglePos(NodePos, vSDFField, vCellData, vCLDGridInfo, Dist);
	if (Dist > 2.0 * vSupportRadius)
	{
		voGridNodeValue[Index] = 0.0;
		return;
	}

	SAABB IntegrandDomain;
	IntegrandDomain.Min = Vector3(-vSupportRadius, -vSupportRadius, -vSupportRadius);
	IntegrandDomain.Max = Vector3(vSupportRadius, vSupportRadius, vSupportRadius);

	voGridNodeValue[Index] = 0.8 * integrateGaussQuadrature(
		IntegrandDomain, 
		vSupportRadius, 
		NodePos, 
		vSDFField, 
		vCellData, 
		vCLDGridInfo
	);
}

__global__ void sendSDFDataToCLDGrid
(
	SCLDGridInfo vCLDGridInfo,
	cudaTextureObject_t vSDFTexture,
	Vector3 vSDFMin,
	Vector3 vSDFInvCellSize,
	const UInt* vCellData,
	Real* vioGridNodeValue,
	bool vIsInvOutside
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vCLDGridInfo.TotalNodeNum) return;

	Vector3 NodePos = vCLDGridInfo.indexToNodePosition(Index);

	Vector3ui NodeCoord = castToVector3ui((NodePos - vSDFMin) * vSDFInvCellSize);
	Real SDFValue = (Real)tex3D<float>(vSDFTexture, NodeCoord.x, NodeCoord.y, NodeCoord.z);
	if (vIsInvOutside) SDFValue *= -1;
	vioGridNodeValue[Index] = SDFValue;
}

void sendSDFDataToCLDGridInvoker
(
	SCLDGridInfo vCLDGridInfo,
	Real* voGridNodeValue,
	const UInt* vCellData, 
	cudaTextureObject_t vSDFTexture,
	bool vIsInvOutside,
	Vector3 vSDFMin,
	Vector3 vSDFInvCellSize
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vCLDGridInfo.TotalNodeNum, BlockSize, GridSize);
	sendSDFDataToCLDGrid  LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vCLDGridInfo,
		vSDFTexture,
		vSDFMin,
		vSDFInvCellSize,
		vCellData,
		voGridNodeValue,
		vIsInvOutside
	);
	CUDA_SYN
}

void generateVolumeDataInvoker
(
	SCLDGridInfo vCLDGridInfo,
	Real* voGridNodeValue,
	const UInt* vCellData,
	cudaTextureObject_t vSDFTexture,
	const Real* vSDFField,
	Real vSupportRadius
)
{
	//设置光滑核函数
	CCubicKernel smoothKernelCubicCPU;
	smoothKernelCubicCPU.setRadius(vSupportRadius);
	CHECK_CUDA(cudaMemcpyToSymbol(SmoothKernelCubic, &smoothKernelCubicCPU, sizeof(CCubicKernel)));

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vCLDGridInfo.TotalNodeNum, BlockSize, GridSize, 0.25);
	computeVolumeValue LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vCLDGridInfo,
		voGridNodeValue,
		vCellData,
		vSDFField,
		vSupportRadius
	);
	CUDA_SYN
}