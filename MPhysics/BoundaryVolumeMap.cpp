#include "BoundaryVolumeMap.h"
#include "BoundaryVolumeMapKernel.cuh"
#include "ThrustWapper.cuh"
#include "CudaContextManager.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

void __processMeshNode
(
	const aiNode * vNode, 
	const aiScene * vSceneObjPtr,
	SBoundaryTransform vInitTranform,
	vector<float>& voVertex, 
	vector<float>& voNormal, 
	UInt& voTriangleCount,
	SAABB& voMeshAABB
)
{
	for (size_t i = 0; i < vNode->mNumMeshes; ++i)
	{
		const aiMesh* MeshPtr = vSceneObjPtr->mMeshes[vNode->mMeshes[i]];
		voTriangleCount += MeshPtr->mNumFaces;

		for (UInt j = 0; j < MeshPtr->mNumFaces; j++)
		{
			aiFace Face = MeshPtr->mFaces[j];
			for (int k = 0; k < Face.mNumIndices; k++)
			{
				UInt VertexIndex = Face.mIndices[k];
				Vector3 CurrVertex = Vector3(
					MeshPtr->mVertices[VertexIndex].x,
					MeshPtr->mVertices[VertexIndex].y,
					MeshPtr->mVertices[VertexIndex].z
				);
				Vector3 CurrNormal = Vector3(
					MeshPtr->mNormals[VertexIndex].x,
					MeshPtr->mNormals[VertexIndex].y,
					MeshPtr->mNormals[VertexIndex].z
				);

				CurrVertex = vInitTranform.Rotaion * CurrVertex;
				CurrVertex *= vInitTranform.Scale;
				CurrVertex += vInitTranform.Pos;

				CurrNormal = vInitTranform.Rotaion * CurrNormal;

				voVertex.push_back(CurrVertex.x);
				voVertex.push_back(CurrVertex.y);
				voVertex.push_back(CurrVertex.z);
				voNormal.push_back(CurrNormal.x);
				voNormal.push_back(CurrNormal.y);
				voNormal.push_back(CurrNormal.z);
				voMeshAABB.update(CurrVertex);
			}
		}
	}
	for (size_t i = 0; i < vNode->mNumChildren; ++i)
	{
		__processMeshNode(vNode->mChildren[i], vSceneObjPtr, vInitTranform, voVertex, voNormal, voTriangleCount, voMeshAABB);
	}
}

CRigidBodyBoundaryVolumeMap::CRigidBodyBoundaryVolumeMap() {}

CRigidBodyBoundaryVolumeMap::CRigidBodyBoundaryVolumeMap(const std::string & vCachePath)
{
	m_VolumeMap = make_unique<CCubicLagrangeDiscreteGrid>();
	m_VolumeMap->load(vCachePath);
}

CRigidBodyBoundaryVolumeMap::~CRigidBodyBoundaryVolumeMap(){}

void CRigidBodyBoundaryVolumeMap::bindBoundaryMesh
(
	const vector<std::string> & vTriangleMeshFilePath, 
	const vector<SBoundaryTransform> & vBoundaryInitTransform, 
	Vector3ui vVolumeMapRes,
	Real vSupportRadius,
	bool vIsInvOutside,
	bool vIsLogSDFResult,
	const char* vCachePath
)
{
	_ASSERT(m_SDFCuda3dArray == nullptr);
	_ASSERT(m_CUDADX11SDFResource == nullptr);
	_ASSERT(m_VolumeMap == nullptr);

	//读取模型并确定AABBBox
	std::vector<float> VertexData;
	std::vector<float> NormalData;
	UInt TriangleCount = 0;
	m_Domain.Max = Vector3(-REAL_MAX, -REAL_MAX, -REAL_MAX);
	m_Domain.Min = Vector3(REAL_MAX, REAL_MAX, REAL_MAX);

	for (UInt i = 0; i < vTriangleMeshFilePath.size(); i++)
	{
		Assimp::Importer Importer;
		const aiScene* SceneObjPtr = Importer.ReadFile(vTriangleMeshFilePath[i], aiProcess_Triangulate | aiProcess_GenNormals);
		__processMeshNode(SceneObjPtr->mRootNode, SceneObjPtr, vBoundaryInitTransform[i], VertexData, NormalData, TriangleCount, m_Domain);
	}

	m_Domain.Max += (8.0*vSupportRadius + m_MapThickness);
	m_Domain.Min -= (8.0*vSupportRadius + m_MapThickness);
	m_Resolution = vVolumeMapRes;
	Vector3 DomainCellLength = m_Domain.getDia() / castToVector3(m_Resolution);

	//用外部的DX11接口计算出SDF，然后将存有SDF数据的DX11资源绑定到CUDA指针上
	m_SDFDomain = m_Domain;
	Vector3 SDFDomainCellLength = DomainCellLength / 3.0;
	m_SDFDomain.Min -= SDFDomainCellLength / 2.0;
	m_SDFDomain.Max += SDFDomainCellLength / 2.0;
	m_SDFResolution = castToVector3ui(m_SDFDomain.getDia() / SDFDomainCellLength);
	setSignRayCountGPU(32);
	SSDF SDFDX11Data = __computeDX11SDF(VertexData, NormalData, TriangleCount, m_SDFDomain, m_SDFResolution);

	if(vIsLogSDFResult) logSDFData();
	__attchDX11SDFDataToCUDA(SDFDX11Data);

	//调用Kernel函数，传递SDF数据，并计算出物体空间下StaticMesh的VolumeMap数据
	m_VolumeMap = make_unique<CCubicLagrangeDiscreteGrid>(m_Domain, m_Resolution, 2);
	ContinuousFunction SendSDFDataToCLDGridFunc = std::bind(
		sendSDFDataToCLDGridInvoker,
		std::placeholders::_1, 
		std::placeholders::_2, 
		std::placeholders::_3, 
		m_SDF3dTexture,
		vIsInvOutside,
		m_SDFDomain.Min,
		castToVector3(m_SDFResolution) / m_SDFDomain.getDia()
	);
	m_VolumeMap->setNodeValue(0, SendSDFDataToCLDGridFunc);

	ContinuousFunction VolumeDataGenerateFunc = std::bind(
		generateVolumeDataInvoker,
		std::placeholders::_1,
		std::placeholders::_2,
		std::placeholders::_3,
		m_SDF3dTexture,
		getRawDevicePointerReal(m_VolumeMap->getField(0)),
		vSupportRadius
	);
	m_VolumeMap->setNodeValue(1, VolumeDataGenerateFunc);

	if (vCachePath != nullptr)
		m_VolumeMap->store(vCachePath);

	__unattchDX11SDFDataToCUDA();
}

void CRigidBodyBoundaryVolumeMap::doInfluenceToParticle(CParticleGroup& vioTarget)
{
	_ASSERT(m_VolumeMap != nullptr);

	for (UInt i = 0; i < m_InstanceCount; i++)
	{
		SBoundaryTransform Transform = m_InstanceTansforms[i];

		if (getDeviceVectorSize(m_BoundaryDistCache[i]) != vioTarget.getSize())
		{
			resizeDeviceVector(m_BoundaryDistCache[i], vioTarget.getSize());
		}
		if (getDeviceVectorSize(m_BoundaryVolumeCache[i]) != vioTarget.getSize())
		{
			resizeDeviceVector(m_BoundaryVolumeCache[i], vioTarget.getSize());
		}
		if (getDeviceVectorSize(m_BoundaryClosestPosCache[i]) != vioTarget.getSize() * 3)
		{
			resizeDeviceVector(m_BoundaryClosestPosCache[i], vioTarget.getSize() * 3);
		}

		m_VolumeMap->interpolateLargeDataSet
		(
			0,
			vioTarget.getSize(),
			vioTarget.getConstParticlePosGPUPtr(),
			getRawDevicePointerReal(m_BoundaryDistCache[i]),
			getRawDevicePointerReal(m_BoundaryClosestPosCache[i]),/*将法线临时存在这里*/
			Transform.Pos,
			Transform.Rotaion
		);
		m_VolumeMap->interpolateLargeDataSet
		(
			1,
			vioTarget.getSize(),
			vioTarget.getConstParticlePosGPUPtr(),
			getRawDevicePointerReal(m_BoundaryVolumeCache[i]),
			nullptr,
			Transform.Pos,
			Transform.Rotaion
		);

		queryVolumeAndClosestPointInvoker
		(
			getReadOnlyRawDevicePointer(m_BoundaryDistCache[i]),
			vioTarget.getParticlePosGPUPtr(),
			vioTarget.getParticleVelGPUPtr(),
			vioTarget.getSize(),
			vioTarget.getParticleSupportRadius(),
			vioTarget.getParticleRadius(),
			getRawDevicePointerReal(m_BoundaryVolumeCache[i]),
			getRawDevicePointerReal(m_BoundaryClosestPosCache[i]),
			Transform.Pos,
			Transform.Rotaion,
			Transform.InvRotation
		);
	}
}

void CRigidBodyBoundaryVolumeMap::logVolumeData()
{
	_ASSERT(m_VolumeMap != nullptr);
	m_VolumeMap->logFieldData(1);
}

unsigned int CRigidBodyBoundaryVolumeMap::addInstance
(
	SMatrix3x3 vRotation,
	Vector3 vPos
)
{
	SBoundaryTransform Transform;
	Transform.Pos = vPos;
	Transform.Rotaion = vRotation;
	Transform.InvRotation = vRotation.getInv();

	m_InstanceTansforms.push_back(Transform);

	m_BoundaryDistCache.push_back(thrust::device_vector<Real>());
	m_BoundaryVolumeCache.push_back(thrust::device_vector<Real>());
	m_BoundaryClosestPosCache.push_back(thrust::device_vector<Real>());

	m_InstanceCount++;
	return m_InstanceCount - 1;
}

void CRigidBodyBoundaryVolumeMap::transformInstance(UInt vInstanceIndex, SMatrix3x3 vRotation, Vector3 vPos)
{
	m_InstanceTansforms[vInstanceIndex].Pos = vPos;
	m_InstanceTansforms[vInstanceIndex].Rotaion = vRotation;
	m_InstanceTansforms[vInstanceIndex].InvRotation = vRotation.getInv();
}

const SAABB & CRigidBodyBoundaryVolumeMap::getDomain() const
{
	return m_Domain;
}

const Vector3ui& CRigidBodyBoundaryVolumeMap::getResolution() const
{
	return m_Resolution;
}

const SAABB& CRigidBodyBoundaryVolumeMap::getSDFDomain() const
{
	return m_SDFDomain;
}

const Vector3ui& CRigidBodyBoundaryVolumeMap::getSDFRes() const
{
	return m_SDFResolution;
}

UInt CRigidBodyBoundaryVolumeMap::getInstanceCount() const
{
	return m_InstanceCount;
}

const thrust::device_vector<Real>& CRigidBodyBoundaryVolumeMap::getBoundaryVolumeCache(UInt vInstanceIndex) const
{
	return m_BoundaryVolumeCache[vInstanceIndex];
}

const thrust::device_vector<Real>& CRigidBodyBoundaryVolumeMap::getBoundaryDistCache(UInt vInstanceIndex) const
{
	return m_BoundaryDistCache[vInstanceIndex];
}

const thrust::device_vector<Real>& CRigidBodyBoundaryVolumeMap::getBoundaryClosestPosCache(UInt vInstanceIndex) const
{
	return m_BoundaryClosestPosCache[vInstanceIndex];
}

const std::shared_ptr<CCubicLagrangeDiscreteGrid>& CRigidBodyBoundaryVolumeMap::getVolumeMap() const
{
	return m_VolumeMap;
}

SSDF CRigidBodyBoundaryVolumeMap::__computeDX11SDF( std::vector<float> vVertex, std::vector<float> vNormal, UInt vTriangleCount, SAABB vSDFDomain, Vector3ui vSDFRes)
{
	return generateSDFGPU
	(
		vVertex.data(), 
		vNormal.data(), 
		vTriangleCount,
		vSDFDomain.Min.x, vSDFDomain.Min.y, vSDFDomain.Min.z,
		vSDFDomain.Max.x, vSDFDomain.Max.y, vSDFDomain.Max.z,
		vSDFRes.x, vSDFRes.y, vSDFRes.z
	);
}

void CRigidBodyBoundaryVolumeMap::__attchDX11SDFDataToCUDA(SSDF vDX11SDFData)
{
	CHECK_CUDA(cudaGraphicsD3D11RegisterResource(
		&m_CUDADX11SDFResource,
		static_cast<ID3D11Resource*>(vDX11SDFData.tex),
		cudaGraphicsRegisterFlagsNone
	));

	CHECK_CUDA(cudaGraphicsMapResources(
		1,
		&m_CUDADX11SDFResource
	));

	CHECK_CUDA(cudaGraphicsResourceGetMappedMipmappedArray(&m_SDFCuda3dArray, m_CUDADX11SDFResource));

	cudaResourceDesc SDF3dResourceDesc;
	memset(&SDF3dResourceDesc, 0, sizeof(cudaResourceDesc));
	SDF3dResourceDesc.resType = cudaResourceTypeMipmappedArray;
	SDF3dResourceDesc.res.mipmap.mipmap = m_SDFCuda3dArray;

	cudaTextureDesc SDF3dTextureDesc;
	memset(&SDF3dTextureDesc, 0, sizeof(cudaTextureDesc));

	SDF3dTextureDesc.normalizedCoords = false;
	SDF3dTextureDesc.filterMode = cudaFilterModePoint;
	SDF3dTextureDesc.addressMode[0] = cudaAddressModeClamp;
	SDF3dTextureDesc.addressMode[1] = cudaAddressModeClamp;
	SDF3dTextureDesc.addressMode[2] = cudaAddressModeClamp;
	SDF3dTextureDesc.readMode = cudaReadModeElementType;

	CHECK_CUDA(cudaCreateTextureObject(&m_SDF3dTexture, &SDF3dResourceDesc, &SDF3dTextureDesc, NULL));
}

void CRigidBodyBoundaryVolumeMap::__unattchDX11SDFDataToCUDA()
{
	CHECK_CUDA(cudaDeviceSynchronize());
	if (m_CUDADX11SDFResource != nullptr)
	{
		CHECK_CUDA(cudaGraphicsUnmapResources(
			1,
			&m_CUDADX11SDFResource
		));
	}

	if (m_SDFCuda3dArray != nullptr)
	{
		CHECK_CUDA(cudaFreeMipmappedArray(m_SDFCuda3dArray));
	}

	CHECK_CUDA(cudaDestroyTextureObject(m_SDF3dTexture));
}
