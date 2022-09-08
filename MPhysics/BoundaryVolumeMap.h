#pragma once
#include "Common.h"
#include "../ThirdParty/sdf/SDFInterface.h";

#include "CubicLagrangeDiscreteGrid.h"
#include "Particle.h"
#include <cuda_d3d11_interop.h>

struct SBoundaryTransform
{
	SMatrix3x3 Rotaion;
	SMatrix3x3 InvRotation;
	Vector3 Pos = Vector3(0, 0, 0);
	Vector3 Scale = Vector3(1, 1, 1);
};

class CRigidBodyBoundaryVolumeMap
{
public:
	CRigidBodyBoundaryVolumeMap();
	CRigidBodyBoundaryVolumeMap(const std::string& vCachePath);
	~CRigidBodyBoundaryVolumeMap();

	void bindBoundaryMesh
	(
		const vector<std::string> & vTriangleMeshFilePath,
		const vector<SBoundaryTransform> & vBoundaryInitTransform,
		Vector3ui vVolumeMapRes,
		Real vSupportRadius,
		bool vIsInvOutside,
		bool vIsLogSDFResult,
		const char* vCachePath
	);

	void doInfluenceToParticle(CParticleGroup& vioTarget);

	void logVolumeData();

	unsigned int addInstance
	(
		SMatrix3x3 vRotation,
		Vector3 vPos
	);

	void transformInstance
	(
		UInt vInstanceIndex,
		SMatrix3x3 vRotation,
		Vector3 vPos
	);

	const SAABB& getDomain() const;
	const Vector3ui& getResolution() const;

	const SAABB& getSDFDomain() const;
	const Vector3ui& getSDFRes() const;

	UInt getInstanceCount() const;

	const thrust::device_vector<Real>& getBoundaryVolumeCache(UInt vInstanceIndex) const;
	const thrust::device_vector<Real>& getBoundaryDistCache(UInt vInstanceIndex) const;
	const thrust::device_vector<Real>& getBoundaryClosestPosCache(UInt vInstanceIndex) const;

	const std::shared_ptr<CCubicLagrangeDiscreteGrid>& getVolumeMap() const;

#ifdef TEST
	thrust::device_vector<Real>& getField(UInt vFieldIndex) { return m_VolumeMap->getField(vFieldIndex); }
	thrust::device_vector<UInt>& getCell() { return m_VolumeMap->getCell(); }
	SCLDGridInfo getCLDGridInfo() const { return m_VolumeMap->getCLDGridInfo(); }
	cudaMipmappedArray_t getExtSDFCuda3dArray() const { return m_SDFCuda3dArray; }
	cudaTextureObject_t getExtSDFCuda3dTexture() const { return m_SDF3dTexture; }
#endif

private:
	UInt m_InstanceCount = 0;
	vector<SBoundaryTransform> m_InstanceTansforms;
	vector<thrust::device_vector<Real>> m_BoundaryVolumeCache;
	vector<thrust::device_vector<Real>> m_BoundaryDistCache;
	vector<thrust::device_vector<Real>> m_BoundaryClosestPosCache;

	SAABB m_Domain;
	Vector3ui m_Resolution;

	SAABB m_SDFDomain;
	Vector3ui m_SDFResolution;

	Real m_MapThickness = 0.0;

	std::shared_ptr<CCubicLagrangeDiscreteGrid> m_VolumeMap = nullptr;

	cudaTextureObject_t m_SDF3dTexture;
	cudaMipmappedArray_t m_SDFCuda3dArray = nullptr;
	cudaGraphicsResource_t m_CUDADX11SDFResource = nullptr;

	SSDF __computeDX11SDF
	(
		std::vector<float> vVertex,
		std::vector<float> vNormal,
		UInt vTriangleCount,
		SAABB vSDFDomain, 
		Vector3ui vSDFRes
	);
	void __attchDX11SDFDataToCUDA(SSDF vDX11SDFData);
	void __unattchDX11SDFDataToCUDA();
};