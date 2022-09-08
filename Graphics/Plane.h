#pragma once
#include "Mesh.h"

namespace graphics
{
	class CPlane : public CMesh
	{
	public:
		CPlane(float vEdgeLength) : CPlane(vEdgeLength, vEdgeLength) {}
		CPlane(float vWidth, float vHeight)
		{
			float HalfWidth = vWidth / 2.0f;
			float HalfHeight = vHeight / 2.0f;
			std::vector<SVertex> Vertices;
			std::vector<unsigned int> Indices;
			SVertex Current;
			Current.Normal = glm::vec3(0.0f, 1.0f, 0.0f);
			float TexcoordScale = vWidth / 8.0f;
			// Point0
			Current.Position = glm::vec3(-HalfWidth, 0.0f, -HalfHeight);
			Current.TexCoords = glm::vec2(0.0f, TexcoordScale);
			Vertices.push_back(Current);
			// Point1
			Current.Position = glm::vec3(-HalfWidth, 0.0f, +HalfHeight);
			Current.TexCoords = glm::vec2(0.0f, 0.0f);
			Vertices.push_back(Current);
			// Point2
			Current.Position = glm::vec3(+HalfWidth, 0.0f, +HalfHeight);
			Current.TexCoords = glm::vec2(TexcoordScale, 0.0f);
			Vertices.push_back(Current);
			// Point3
			Current.Position = glm::vec3(+HalfWidth, 0.0f, -HalfHeight);
			Current.TexCoords = glm::vec2(TexcoordScale, TexcoordScale);
			Vertices.push_back(Current);

			Indices = { 0,1,2, 0,2,3 };

			_setupMesh(Vertices, Indices);
		}

		virtual ~CPlane() = default;
	};
}