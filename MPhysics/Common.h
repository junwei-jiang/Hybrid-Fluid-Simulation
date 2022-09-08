#pragma once

//必须放在下面两个namespace前面
#include "EnumType.h"
#include "MMathExtra.h"
#include "CudaMathTools.h"

#define NOMINMAX
#include <windows.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_set>
#include <memory>
#include <string>
#include <fstream>
#include <array>
#include <functional>
#include <filesystem>
using namespace std;

#include <boost/archive/binary_iarchive.hpp> //文本格式输入存档
#include <boost/archive/binary_oarchive.hpp> //文本格式输出存档
#include <boost/archive/text_iarchive.hpp> //文本格式输入存档
#include <boost/archive/text_oarchive.hpp> //文本格式输出存档
#include <boost/serialization/vector.hpp>  //vector的序列化实现头文件
using namespace boost::archive;//打开名称空间

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace boost {
	namespace serialization {

		template<class Archive>
		void serialize(Archive & ar, Vector3 & d, const unsigned int version)
		{
			ar & d.x;
			ar & d.y;
			ar & d.z;
		}

		template<class Archive>
		void serialize(Archive & ar, Vector3ui & d, const unsigned int version)
		{
			ar & d.x;
			ar & d.y;
			ar & d.z;
		}

		template<class Archive>
		void serialize(Archive & ar, Vector3i & d, const unsigned int version)
		{
			ar & d.x;
			ar & d.y;
			ar & d.z;
		}

	}
}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/count.h>

#ifdef DOUBLE_REAL
#define CUDA_REAL_TYPE CUDA_R_64F
#define cublasCopy cublasDcopy
#define cublasScal cublasDscal
#define cublasAxpy cublasDaxpy
#define cublasNrm2 cublasDnrm2
#define cublasDot cublasDdot
#define cublasMax cublasIdamax
#define cublasMin cublasIdamin
#define curandUniform curand_uniform
#define GRID_SOLVER_EPSILON 1e-4
#else
#define CUDA_REAL_TYPE CUDA_R_32F
#define cublasCopy cublasScopy
#define cublasScal cublasSscal
#define cublasAxpy cublasSaxpy
#define cublasNrm2 cublasSnrm2
#define cublasDot cublasSdot
#define cublasMax cublasIsamax
#define cublasMin cublasIsamin
#define curandUniform curand_uniform_double
#define GRID_SOLVER_EPSILON 1e-4
#endif // DOUBLE_REAL

#define cudaMallocM cudaMalloc

#define LANCH_PARAM_1D_GB(GridNum, BlockNum) <<<GridNum, BlockNum>>>
#define LANCH_PARAM_1D_GBS(GridNum, BlockNum, ShareMemSize) <<<GridNum, BlockNum, ShareMemSize>>>
#define LANCH_PARAM_1D_GBSS(GridNum, BlockNum, ShareMemSize, Stream) <<<GridNum, BlockNum, ShareMemSize, Stream>>>

#define CUDA_SYN //CHECK_CUDA(cudaDeviceSynchronize());

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
		cout << "CUDA API failed " << #func << " at line " << __LINE__		   \
		<< " with error: " << cudaGetErrorString(status) << " "			       \
		<< status << endl;													   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);										   \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
		cout << "CUSPARSE API failed " << #func << " at line " << __LINE__     \
		<< " with error: " << cusparseGetErrorString(status) << " "			   \
		<< status << endl;													   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                   \
{                                                                            \
    cublasStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                 \
        cout << "CUBLAS API failed " << #func << " at line " << __LINE__     \
		<< " with error: " << to_string(status) << " " << status << endl;	 \
    }                                                                        \
}

#define CHECK_CUDA_LAST_ERROR												   \
{                                                                              \
    cudaError_t status = cudaGetLastError();                                   \
    if (status != cudaSuccess) {                                               \
		cout << "CUDA API ERROR FOUND " << " at line " << __LINE__			   \
		<< " with error: " << cudaGetErrorString(status) << " "			       \
		<< status << endl;													   \
    }                                                                          \
}

struct SAABB
{
	Vector3 Min = Vector3(REAL_MAX, REAL_MAX, REAL_MAX);
	Vector3 Max = Vector3(-REAL_MAX, -REAL_MAX, -REAL_MAX);

	__host__ __device__ SAABB()
	{
		Min = Vector3(-REAL_MAX, -REAL_MAX, -REAL_MAX);
		Max = Vector3(REAL_MAX, REAL_MAX, REAL_MAX);
	}	
	
	__host__ __device__ SAABB(const Vector3& vMin, const Vector3& vMax)
	{
		Min = vMin;
		Max = vMax;
	}

	__host__ __device__ bool contains(Vector3 vX) const
	{
		return 
			(vX.x >= Min.x &&
			vX.y >= Min.y &&
			vX.z >= Min.z &&
			vX.x <= Max.x &&
			vX.y <= Max.y &&
			vX.z <= Max.z);
	}

	__host__ __device__ void update(Vector3 vX)
	{
		if (vX.x > Max.x) Max.x = vX.x;
		if (vX.y > Max.y) Max.y = vX.y;
		if (vX.z > Max.z) Max.z = vX.z;

		if (vX.x < Min.x) Min.x = vX.x;
		if (vX.y < Min.y) Min.y = vX.y;
		if (vX.z < Min.z) Min.z = vX.z;
	}

	__host__ __device__ Vector3 getDia() const
	{
		return Max - Min;
	}

	__host__ __device__ Real getDistance(Vector3 vX) const
	{
		if (!this->contains(vX))
		{
			Vector3 Result = vX;
			if (vX.x < Min.x) Result.x = Min.x;
			if (vX.y < Min.y) Result.y = Min.y;
			if (vX.z < Min.z) Result.z = Min.z;

			if (vX.x > Max.x) Result.x = Max.x;
			if (vX.y > Max.y) Result.y = Max.y;
			if (vX.z > Max.z) Result.z = Max.z;

			return length(Result - vX);
		}
		else
		{
			Real DisX = thrust::min(abs(vX.x - Min.x), abs(vX.x - Max.x));
			Real DisY = thrust::min(abs(vX.y - Min.y), abs(vX.y - Max.y));
			Real DisZ = thrust::min(abs(vX.z - Min.z), abs(vX.z - Max.z));

			return thrust::min(thrust::min(DisX, DisY), DisZ);
		}
	}

	__host__ __device__ void lerp(SAABB& vioAABB)
	{
		if (vioAABB.Min.x < Min.x) vioAABB.Min.x = Min.x;
		if (vioAABB.Min.y < Min.y) vioAABB.Min.y = Min.y;
		if (vioAABB.Min.z < Min.z) vioAABB.Min.z = Min.z;

		if (vioAABB.Max.x > Max.x) vioAABB.Max.x = Max.x;
		if (vioAABB.Max.y > Max.y) vioAABB.Max.y = Max.y;
		if (vioAABB.Max.z > Max.z) vioAABB.Max.z = Max.z;
	}

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & Min;
		ar & Max;
	}
};

#define PFDebugTarget 25
#define MAXNeighborCount 64
#define FluidSurfaceDensity 0.2
#define FluidDomainValue 500
#define SolidDomainValue 400
#define AirDomainValue 300
#define BoundaryOffset 0.5

#define convert3DIndexToLiner(vIndex, vRes) vIndex.z * vRes.x * vRes.y + vIndex.y * vRes.x + vIndex.x

inline void HighLightLog(string vTitle, string vMessage)
{
#ifdef WIN64
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_GREEN);
	cout << "[" << vTitle <<"] ";
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
#else
	cout << "[" << vTitle << "] ";
#endif // WIN64
	cout << vMessage << endl;
}