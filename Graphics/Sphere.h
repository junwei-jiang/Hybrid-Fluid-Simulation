#pragma once
#include "Mesh.h"
#include <corecrt_math_defines.h>

namespace graphics
{
	class CSphere : public CMesh
	{
	public:
		CSphere() = default;
		CSphere(float vRadius, int vNumLatitudeLine = 20, int vNumLongitudeLine = 20)
		{
			std::vector<SVertex> Vertices;
			std::vector<unsigned int> Indices;

			float DeltaRingAngle = static_cast<float>(M_PI / vNumLatitudeLine);
			float DeltaSegAngle = static_cast<float>(M_PI * 2 / vNumLongitudeLine);
			unsigned short VerticeIndex = 0;

			// Generate the group of rings for the sphere.
			for (int i = 0; i <= vNumLatitudeLine; i++)
			{
				float RCoord = static_cast<float>(vRadius) * sinf(i * DeltaRingAngle);
				float YRCoord = static_cast<float>(vRadius) * cosf(i * DeltaRingAngle);

				// Generate the group of segments for the current ring.
				for (int k = 0; k <= vNumLongitudeLine; k++)
				{
					float XCoord = RCoord * sinf(k * DeltaSegAngle);
					float ZCoord = RCoord * cosf(k * DeltaSegAngle);

					glm::vec3 Normal = glm::normalize(glm::vec3(XCoord, YRCoord, ZCoord));

					Vertices.push_back(SVertex(XCoord, YRCoord, ZCoord, Normal.x, Normal.y, Normal.z,
										static_cast<float>(k) / static_cast<float>(vNumLongitudeLine),
										static_cast<float>(i) / static_cast<float>(vNumLatitudeLine)));
					if (i != vNumLatitudeLine)
					{
						// each vertex (except the last) has six indicies pointing to it.
						Indices.push_back(static_cast<unsigned short>(VerticeIndex + vNumLongitudeLine + 1));
						Indices.push_back(static_cast<unsigned short>(VerticeIndex));
						Indices.push_back(static_cast<unsigned short>(VerticeIndex + vNumLongitudeLine));
						Indices.push_back(static_cast<unsigned short>(VerticeIndex + vNumLongitudeLine + 1));
						Indices.push_back(VerticeIndex + 1);
						Indices.push_back(VerticeIndex);
						VerticeIndex++;
					}
				};
			}

			_setupMesh(Vertices, Indices);
		}

		virtual ~CSphere() = default;
	};
}