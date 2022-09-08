#include "SDFInterface.h"
#include "sdf.h"
#include <vector>
using namespace d3d11_sdf;

SDFGenerator GlobalSDFGenerator;
ID3D11Device*        D3dDevice = nullptr;
ID3D11DeviceContext* D3dDeviceContext = nullptr;
const D3D_FEATURE_LEVEL     FeatureLevel = D3D_FEATURE_LEVEL_11_1;
SDF Result;

void initAgzUtilsDX11Device()
{
	if (FAILED(D3D11CreateDevice(
		nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
		D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT,
		&FeatureLevel, 1, D3D11_SDK_VERSION,
		&D3dDevice, nullptr,
		&D3dDeviceContext)))
	{
		throw std::runtime_error(
			"failed to create d3d1 device & device context");
	}
}

void freeAgzUtilsDX11Device()
{
	if (D3dDevice) { D3dDevice->Release(); D3dDevice = nullptr; }
	if (D3dDeviceContext) { D3dDeviceContext->Release(); D3dDeviceContext = nullptr; }
}

void setSignRayCountGPU(unsigned int vInput)
{
	GlobalSDFGenerator.setSignRayCount(vInput);
}

SSDF generateSDFGPU
(
	const float* vVertices,
	const float* vNormals,
	unsigned int vTriangleCount,
	float vMinX, float vMinY, float vMinZ,
	float vMaxX, float vMaxY, float vMaxZ,
	unsigned int vResX,
	unsigned int vResY,
	unsigned int vResZ
)
{
	if(D3dDevice == nullptr || D3dDeviceContext == nullptr)
		throw std::runtime_error("未初始化 DX11设备！");

	agz::d3d11::device.d3dDevice = D3dDevice;
	agz::d3d11::deviceContext.d3dDeviceContext = D3dDeviceContext;
	AGZ_SCOPE_GUARD({
		agz::d3d11::device.d3dDevice = nullptr;
		agz::d3d11::deviceContext.d3dDeviceContext = nullptr;
		}
	);

	Float3 Lower(vMinX, vMinY, vMinZ);
	Float3 Upper(vMaxX, vMaxY, vMaxZ);
	Int3 Res(vResX, vResY, vResZ);

	std::vector<Float3> Vertices;
	std::vector<Float3> Normals;
	for (unsigned int i = 0; i < vTriangleCount * 3; i++)
	{
		Vertices.push_back(Float3(vVertices[i * 3], vVertices[i * 3 + 1], vVertices[i * 3 + 2]));
	}
	for (unsigned int i = 0; i < vTriangleCount * 3; i++)
	{
		Normals.push_back(Float3(vNormals[i * 3], vNormals[i * 3 + 1], vNormals[i * 3 + 2]));
	}

	Result = GlobalSDFGenerator.generateGPU(Vertices.data(), Normals.data(), vTriangleCount, Lower, Upper, Res);

	SSDF Wapper;
	Wapper.tex = Result.tex.Get();
	Wapper.srv = Result.srv.Get();

	Wapper.vMinX = Result.lower.x;
	Wapper.vMinY = Result.lower.y;
	Wapper.vMinZ = Result.lower.z;

	Wapper.vMaxX = Result.upper.x;
	Wapper.vMaxY = Result.upper.y;
	Wapper.vMaxZ = Result.upper.z;

	Wapper.vResX = Result.res.x;
	Wapper.vResY = Result.res.y;
	Wapper.vResZ = Result.res.z;

	return Wapper;
}

void logSDFData()
{
	if (D3dDevice == nullptr || D3dDeviceContext == nullptr)
		throw std::runtime_error("未初始化 DX11设备！");

	if (Result.tex == nullptr)
		throw std::runtime_error("未生成SDF数据");

	agz::d3d11::device.d3dDevice = D3dDevice;
	agz::d3d11::deviceContext.d3dDeviceContext = D3dDeviceContext;
	AGZ_SCOPE_GUARD({
		agz::d3d11::device.d3dDevice = nullptr;
		agz::d3d11::deviceContext.d3dDeviceContext = nullptr;
		}
	);

	D3D11_TEXTURE3D_DESC texDesc;
	Result.tex->GetDesc(&texDesc);
	texDesc.BindFlags = 0;
	texDesc.Usage = D3D11_USAGE_STAGING;
	texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	auto stagingTex = device.createTex3D(texDesc);

	deviceContext->CopyResource(stagingTex.Get(), Result.tex.Get());

	D3D11_MAPPED_SUBRESOURCE mappedSubrsc;
	deviceContext->Map(stagingTex.Get(), 0, D3D11_MAP_READ, 0, &mappedSubrsc);

	std::vector<float> sdfData;
	sdfData.resize(Result.res.product());
	const size_t rowSize = sizeof(float) * Result.res.x;
	char *copySrc = static_cast<char*>(mappedSubrsc.pData);
	char *copyDst = reinterpret_cast<char*>(sdfData.data());

	for (int z = 0; z < Result.res.z; ++z)
	{
		for (int y = 0; y < Result.res.y; ++y)
		{
			auto row = copySrc + mappedSubrsc.RowPitch * y;
			std::memcpy(copyDst, row, rowSize);
			copyDst += rowSize;
		}
		copySrc += mappedSubrsc.DepthPitch;
	}

	for (int z = 0; z < Result.res.z; ++z)
	{
		for (int y = 0; y < Result.res.y; ++y)
		{
			for (int x = 0; x < Result.res.x; x++)
				std::cout << sdfData[Result.res.x * Result.res.y * z + Result.res.x * y + x] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}
