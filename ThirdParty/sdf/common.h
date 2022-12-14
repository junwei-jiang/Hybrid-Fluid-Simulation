#pragma once

#include <agz-utils\graphics_api.h>

#define D3D11_SDF_BEGIN namespace d3d11_sdf {
#define D3D11_SDF_END   }

D3D11_SDF_BEGIN

using namespace agz::d3d11;

inline float lerp(float a, float b, float f)
{
	return a + f * (b - a);
}

D3D11_SDF_END
