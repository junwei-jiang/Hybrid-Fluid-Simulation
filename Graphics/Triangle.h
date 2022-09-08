#pragma once
#include "Mesh.h"

namespace graphics
{
	class CTriangle : public CMesh
	{
	public:
		CTriangle(const glm::vec3& vPoint1, const glm::vec3& vPoint2, const glm::vec3& vPoint3)
		{
			std::vector<SVertex> Vertices;
			std::vector<unsigned int> Indices;
			SVertex Current;
			Current.Normal = glm::normalize(glm::cross(vPoint2 - vPoint1, vPoint3 - vPoint1));
			// Point1
			Current.Position = vPoint1;
			Current.TexCoords = glm::vec2(0.5f, 1.0f);
			Vertices.push_back(Current);
			Indices.push_back(0);
			// Point2
			Current.Position = vPoint2;
			Current.TexCoords = glm::vec2(0.0f, 0.0f);
			Vertices.push_back(Current);
			Indices.push_back(1);
			// Point3
			Current.Position = vPoint3;
			Current.TexCoords = glm::vec2(1.0f, 0.0f);
			Vertices.push_back(Current);
			Indices.push_back(2);

			_setupMesh(Vertices, Indices);
		}

		virtual ~CTriangle() = default;
	};
}