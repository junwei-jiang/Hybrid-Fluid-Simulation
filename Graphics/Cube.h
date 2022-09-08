#pragma once
#include "Mesh.h"

namespace graphics
{
	class CCube : public CMesh
	{
	public:
		CCube(float vEdgeLength) : CCube(vEdgeLength, vEdgeLength, vEdgeLength) {}
		CCube(float vWidth, float vHeight, float vDepth)
		{
			float HalfWidth = vWidth * 0.5f;
			float HalfHeight = vHeight * 0.5f;
			float HalfDepth = vDepth * 0.5f;

			std::vector<SVertex>  Vertices =
			{
				SVertex(-HalfWidth, -HalfHeight, -HalfDepth, 0.f, 0.f, -1.f, 0.f, 1.f),
				SVertex(-HalfWidth, +HalfHeight, -HalfDepth, 0.f, 0.f, -1.f, 0.f, 0.f),
				SVertex(+HalfWidth, +HalfHeight, -HalfDepth, 0.f, 0.f, -1.f, 1.f, 0.f),
				SVertex(+HalfWidth, -HalfHeight, -HalfDepth, 0.f, 0.f, -1.f, 1.f, 1.f),

				SVertex(-HalfWidth, -HalfHeight, +HalfDepth, -1.f, 0.f, 0.f, 0.f, 1.f),
				SVertex(-HalfWidth, +HalfHeight, +HalfDepth, -1.f, 0.f, 0.f, 0.f, 0.f),
				SVertex(-HalfWidth, +HalfHeight, -HalfDepth, -1.f, 0.f, 0.f, 1.f, 0.f),
				SVertex(-HalfWidth, -HalfHeight, -HalfDepth, -1.f, 0.f, 0.f, 1.f, 1.f),

				SVertex(+HalfWidth, -HalfHeight, +HalfDepth, 0.f, 0.f, 1.f, 0.f, 1.f),
				SVertex(+HalfWidth, +HalfHeight, +HalfDepth, 0.f, 0.f, 1.f, 0.f, 0.f),
				SVertex(-HalfWidth, +HalfHeight, +HalfDepth, 0.f, 0.f, 1.f, 1.f, 0.f),
				SVertex(-HalfWidth, -HalfHeight, +HalfDepth, 0.f, 0.f, 1.f, 1.f, 1.f),
				
				SVertex(+HalfWidth, -HalfHeight, -HalfDepth, 1.f, 0.f, 0.f, 0.f, 1.f),
				SVertex(+HalfWidth, +HalfHeight, -HalfDepth, 1.f, 0.f, 0.f, 0.f, 0.f),
				SVertex(+HalfWidth, +HalfHeight, +HalfDepth, 1.f, 0.f, 0.f, 1.f, 0.f),
				SVertex(+HalfWidth, -HalfHeight, +HalfDepth, 1.f, 0.f, 0.f, 1.f, 1.f),
				
				SVertex(-HalfWidth, +HalfHeight, -HalfDepth, 0.f, 1.f, 0.f, 0.f, 1.f),
				SVertex(-HalfWidth, +HalfHeight, +HalfDepth, 0.f, 1.f, 0.f, 0.f, 0.f),
				SVertex(+HalfWidth, +HalfHeight, +HalfDepth, 0.f, 1.f, 0.f, 1.f, 0.f),
				SVertex(+HalfWidth, +HalfHeight, -HalfDepth, 0.f, 1.f, 0.f, 1.f, 1.f),

				SVertex(-HalfWidth, -HalfHeight, +HalfDepth, 0.f, -1.f, 0.f, 0.f, 1.),
				SVertex(-HalfWidth, -HalfHeight, -HalfDepth, 0.f, -1.f, 0.f, 0.f, 0.),
				SVertex(+HalfWidth, -HalfHeight, -HalfDepth, 0.f, -1.f, 0.f, 1.f, 0.),
				SVertex(+HalfWidth, -HalfHeight, +HalfDepth, 0.f, -1.f, 0.f, 1.f, 1.)
			};

			std::vector<unsigned int> Indices =
			{
				// front.
				 0, 1, 2,
				 0, 2, 3,

				 // back.
				 4, 5, 6,
				 4, 6, 7,

				 // left.
				 8, 9,10,
				 8,10,11,

				 // right.
				 12,13,14,
				 12,14,15,

				 // up.
				 16,17,18,
				 16,18,19,

				 // down.
				 20,21,22,
				 20,22,23
			};

			_setupMesh(Vertices, Indices);
		}

		virtual ~CCube() = default;
	};
}