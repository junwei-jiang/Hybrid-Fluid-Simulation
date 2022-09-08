#pragma once
#ifndef SDF_PROJECT
#define SDFAPI extern __declspec(dllimport)
#else
#define SDFAPI extern __declspec(dllexport)
#endif

struct SSDF
{
	void* tex = nullptr;
	void* srv = nullptr;
	float vMinX = 0.0f; float vMinY = 0.0f; float vMinZ = 0.0f;
	float vMaxX = 0.0f; float vMaxY = 0.0f; float vMaxZ = 0.0f;
	unsigned int vResX = 0;
	unsigned int vResY = 0;
	unsigned int vResZ = 0;
};

SDFAPI void initAgzUtilsDX11Device();
SDFAPI void freeAgzUtilsDX11Device();
SDFAPI void setSignRayCountGPU(unsigned int vInput);
SDFAPI SSDF generateSDFGPU
(
	const float* vVertices,
	const float* vNormals,
	unsigned int vTriangleCount,
	float vMinX, float vMinY, float vMinZ,
	float vMaxX, float vMaxY, float vMaxZ,
	unsigned int vResX, 
	unsigned int vResY,
	unsigned int vResZ
);

SDFAPI void logSDFData();