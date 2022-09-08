#pragma once
#include <assimp/scene.h>
#include "Mesh.h"

namespace graphics
{
	class CModel
	{
	public:
		CModel(const std::string& vPath);

		void draw(const CShader& vShader);
		void draw(const CShader* vShader);
		void updateVertexPosition(const std::vector<std::vector<glm::vec3>>& vPos, double vAngleThreshold = -1.0);

		int getMumOfMeshes() const;
		const CMesh& getMeshesAt(int vIndex) const;

	private:
		void __loadModel(const std::string& vPath);
		void __processNode(const aiNode *vNode, const aiScene *vScene);
		CMesh __processMesh(const aiMesh *vMesh, const aiScene *vScene);
		std::vector<STexture> __loadMaterialTextures(const aiMaterial *vMaterial, const aiTextureType& vType, const std::string& vTypeName);
		unsigned int __loadTextureFromFile(const std::string& vPath, const std::string& vDirectory);

		std::vector<CMesh> m_Meshes;
		std::string m_Directory;
		std::vector<STexture> m_TexturesLoaded;
	};
}