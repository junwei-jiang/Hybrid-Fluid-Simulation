#include "BoundaryHelper.h"
#include "FieldMathTool.cuh"
#include "EulerSolverTool.cuh"

#include <cuda_d3d11_interop.h>
#include "../ThirdParty/sdf/SDFInterface.h"

void processMeshNode
(
	const aiNode * vNode,
	const aiScene * vSceneObjPtr,
	vector<float>& voVertex,
	vector<float>& voNormal,
	UInt& voTriangleCount
)
{
	for (size_t i = 0; i < vNode->mNumMeshes; ++i)
	{
		const aiMesh* MeshPtr = vSceneObjPtr->mMeshes[vNode->mMeshes[i]];
		voTriangleCount += MeshPtr->mNumFaces;

		for (UInt j = 0; j < MeshPtr->mNumFaces; j++)
		{
			aiFace Face = MeshPtr->mFaces[j];
			for (UInt k = 0; k < Face.mNumIndices; k++)
			{
				UInt VertexIndex = Face.mIndices[k];
				Vector3 CurrVertex = Vector3(
					MeshPtr->mVertices[VertexIndex].x,
					MeshPtr->mVertices[VertexIndex].y,
					MeshPtr->mVertices[VertexIndex].z
				);
				voVertex.push_back(CurrVertex.x);
				voVertex.push_back(CurrVertex.y);
				voVertex.push_back(CurrVertex.z);
				voNormal.push_back(MeshPtr->mNormals[VertexIndex].x);
				voNormal.push_back(MeshPtr->mNormals[VertexIndex].y);
				voNormal.push_back(MeshPtr->mNormals[VertexIndex].z);
			}
		}
	}
	for (size_t i = 0; i < vNode->mNumChildren; ++i)
	{
		processMeshNode(vNode->mChildren[i], vSceneObjPtr, voVertex, voNormal, voTriangleCount);
	}
}

void generateSDF(string vTriangleMeshFilePath, CCellCenteredScalarField& voSDFField, bool vIsInvSign)
{
	initAgzUtilsDX11Device();

	Assimp::Importer Importer;
	const aiScene* SceneObjPtr = Importer.ReadFile(vTriangleMeshFilePath, aiProcess_Triangulate);

	std::vector<float> VertexData;
	std::vector<float> NormalData;
	Vector3i FieldRes = voSDFField.getResolution();
	Vector3  FiledOrigin = voSDFField.getOrigin();
	Vector3  FiledSpacing = voSDFField.getSpacing();
	Vector3  FieldDomainMin = FiledOrigin;
	Vector3  FieldDomainMax = FiledOrigin + Vector3(FiledSpacing.x * FieldRes.x, FiledSpacing.y * FieldRes.y, FiledSpacing.z * FieldRes.z);
	UInt TriangleCount = 0;

	processMeshNode(SceneObjPtr->mRootNode, SceneObjPtr, VertexData, NormalData, TriangleCount);

	setSignRayCountGPU(12);
	SSDF BoundarySDF = generateSDFGPU
	(
		VertexData.data(),
		NormalData.data(),
		TriangleCount,
		FieldDomainMin.x, FieldDomainMin.y, FieldDomainMin.z,
		FieldDomainMax.x, FieldDomainMax.y, FieldDomainMax.z,
		FieldRes.x, FieldRes.y, FieldRes.z
	);

	//logSDFData();

	cudaTextureObject_t SDF3dTexture;
	cudaMipmappedArray_t SDFCuda3dArray = nullptr;
	cudaGraphicsResource_t CUDADX11SDFResource = nullptr;

	CHECK_CUDA(cudaGraphicsD3D11RegisterResource(
		&CUDADX11SDFResource,
		static_cast<ID3D11Resource*>(BoundarySDF.tex),
		cudaGraphicsRegisterFlagsNone
	));
	CHECK_CUDA(cudaGraphicsMapResources(1, &CUDADX11SDFResource));
	CHECK_CUDA(cudaGraphicsResourceGetMappedMipmappedArray(&SDFCuda3dArray, CUDADX11SDFResource));

	cudaResourceDesc SDF3dResourceDesc;
	memset(&SDF3dResourceDesc, 0, sizeof(cudaResourceDesc));
	SDF3dResourceDesc.resType = cudaResourceTypeMipmappedArray;
	SDF3dResourceDesc.res.mipmap.mipmap = SDFCuda3dArray;

	cudaTextureDesc SDF3dTextureDesc;
	memset(&SDF3dTextureDesc, 0, sizeof(cudaTextureDesc));

	SDF3dTextureDesc.normalizedCoords = false;
	SDF3dTextureDesc.filterMode = cudaFilterModePoint;
	SDF3dTextureDesc.addressMode[0] = cudaAddressModeClamp;
	SDF3dTextureDesc.addressMode[1] = cudaAddressModeClamp;
	SDF3dTextureDesc.addressMode[2] = cudaAddressModeClamp;
	SDF3dTextureDesc.readMode = cudaReadModeElementType;

	CHECK_CUDA(cudaCreateTextureObject(&SDF3dTexture, &SDF3dResourceDesc, &SDF3dTextureDesc, NULL));

	fillSDFFieldInvoker(SDF3dTexture, voSDFField, vIsInvSign);

	if (SDFCuda3dArray != nullptr)
	{
		CHECK_CUDA(cudaFreeMipmappedArray(SDFCuda3dArray));
	}

	if (CUDADX11SDFResource != nullptr)
	{
		CHECK_CUDA(cudaGraphicsUnmapResources(1, &CUDADX11SDFResource));
	}

	freeAgzUtilsDX11Device();
}