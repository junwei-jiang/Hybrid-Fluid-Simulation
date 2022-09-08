#include "Model.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace graphics;

CModel::CModel(const std::string& vPath)
{
	__loadModel(vPath);
}

//****************************************************************************************************
//FUNCTION:
void CModel::draw(const CShader& vShader)
{
	for (unsigned int i = 0; i < m_Meshes.size(); i++)
		m_Meshes[i].draw(vShader);
}

//****************************************************************************************************
//FUNCTION:
void CModel::draw(const CShader* vShader)
{
	draw(*vShader);
}

//****************************************************************************************************
//FUNCTION:
int CModel::getMumOfMeshes() const
{
	_ASSERTE(m_Meshes.size() > 0);
	return m_Meshes.size();
}

//****************************************************************************************************
//FUNCTION:
const CMesh& CModel::getMeshesAt(int vIndex) const
{
	_ASSERTE(m_Meshes.size() > 0);
	return m_Meshes[vIndex];
}

//****************************************************************************************************
//FUNCTION:
void CModel::__loadModel(const std::string& vPath)
{
	// read file via ASSIMP
	Assimp::Importer Importer;
	const aiScene* pScene = Importer.ReadFile(vPath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace | aiProcess_GenNormals);
	// check for errors
	if (!pScene || pScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !pScene->mRootNode) // if is Not Zero
	{
		std::string ErrorString = "ERROR::ASSIMP:: ";
		ErrorString += Importer.GetErrorString();
		throw std::runtime_error(ErrorString);
	}
	// retrieve the directory path of the filepath
	m_Directory = vPath.substr(0, vPath.find_last_of('/'));

	// process ASSIMP's root node recursively
	__processNode(pScene->mRootNode, pScene);
}

//****************************************************************************************************
//FUNCTION:
void CModel::__processNode(const aiNode *vNode, const aiScene *vScene)
{
	// process each mesh located at the current node
	for (unsigned int i = 0; i < vNode->mNumMeshes; i++)
	{
		// the node object only contains indices to index the actual objects in the scene. 
		// the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
		aiMesh* mesh = vScene->mMeshes[vNode->mMeshes[i]];
		m_Meshes.push_back(std::move(__processMesh(mesh, vScene)));
	}
	// after we've processed all of the meshes (if any) we then recursively process each of the children nodes
	for (unsigned int i = 0; i < vNode->mNumChildren; i++)
	{
		__processNode(vNode->mChildren[i], vScene);
	}
}

//****************************************************************************************************
//FUNCTION:
CMesh CModel::__processMesh(const aiMesh *vMesh, const aiScene *vScene)
{
	// data to fill
	std::vector<SVertex> Vertices;
	std::vector<unsigned int> Indices;
	std::vector<STexture> Textures;

	// Walk through each of the mesh's vertices
	for (unsigned int i = 0; i < vMesh->mNumVertices; i++)
	{
		SVertex Vertex;
		glm::vec3 Vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
		// positions
		Vector.x = vMesh->mVertices[i].x;
		Vector.y = vMesh->mVertices[i].y;
		Vector.z = vMesh->mVertices[i].z;
		Vertex.Position = Vector;
		// normals
		Vector.x = vMesh->mNormals[i].x;
		Vector.y = vMesh->mNormals[i].y;
		Vector.z = vMesh->mNormals[i].z;
		Vertex.Normal = Vector;
		// texture coordinates
		if (vMesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
		{
			glm::vec2 Vec;
			// a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
			// use models where a vertex can have multiple texture coordinates so we always take the first set (0).
			Vec.x = vMesh->mTextureCoords[0][i].x;
			Vec.y = vMesh->mTextureCoords[0][i].y;
			Vertex.TexCoords = Vec;
		}
		else
			Vertex.TexCoords = glm::vec2(0.0f, 0.0f);
		// tangent
		Vector.x = vMesh->mTangents[i].x;
		Vector.y = vMesh->mTangents[i].y;
		Vector.z = vMesh->mTangents[i].z;
		Vertex.Tangent = Vector;
		// bitangent
		Vector.x = vMesh->mBitangents[i].x;
		Vector.y = vMesh->mBitangents[i].y;
		Vector.z = vMesh->mBitangents[i].z;
		Vertex.Bitangent = Vector;
		Vertices.push_back(Vertex);
	}
	// now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
	for (unsigned int i = 0; i < vMesh->mNumFaces; i++)
	{
		aiFace Face = vMesh->mFaces[i];
		// retrieve all indices of the face and store them in the indices vector
		for (unsigned int k = 0; k < Face.mNumIndices; k++)
			Indices.push_back(Face.mIndices[k]);
	}
	// process materials
	aiMaterial* Material = vScene->mMaterials[vMesh->mMaterialIndex];
	// we assume a convention for sampler names in the shaders. Each diffuse texture should be named
	// as 'texture_diffuseN' where N is a sequential number ranging from 1 to MAX_SAMPLER_NUMBER. 
	// Same applies to other texture as the following list summarizes:
	// diffuse: texture_diffuseN
	// specular: texture_specularN
	// normal: texture_normalN

	// 1. diffuse maps
	std::vector<STexture> DiffuseMaps = __loadMaterialTextures(Material, aiTextureType_DIFFUSE, "texture_diffuse");
	Textures.insert(Textures.end(), DiffuseMaps.begin(), DiffuseMaps.end());
	// 2. specular maps
	std::vector<STexture> SpecularMaps = __loadMaterialTextures(Material, aiTextureType_SPECULAR, "texture_specular");
	Textures.insert(Textures.end(), SpecularMaps.begin(), SpecularMaps.end());
	// 3. normal maps
	std::vector<STexture> NormalMaps = __loadMaterialTextures(Material, aiTextureType_HEIGHT, "texture_normal");
	Textures.insert(Textures.end(), NormalMaps.begin(), NormalMaps.end());
	// 4. height maps
	std::vector<STexture> HeightMaps = __loadMaterialTextures(Material, aiTextureType_AMBIENT, "texture_height");
	Textures.insert(Textures.end(), HeightMaps.begin(), HeightMaps.end());
	// 5. reflect maps
	std::vector<STexture> ReflectionMaps = __loadMaterialTextures(Material, aiTextureType_AMBIENT, "texture_reflection");
	Textures.insert(Textures.end(), ReflectionMaps.begin(), ReflectionMaps.end());

	// return a mesh object created from the extracted mesh data
	return CMesh(Vertices, Indices, Textures);
}

//****************************************************************************************************
//FUNCTION:
std::vector<STexture> CModel::__loadMaterialTextures(const aiMaterial *vMaterial, const aiTextureType& vType, const std::string& vTypeName)
{
	std::vector<STexture> Textures;
	for (unsigned int i = 0; i < vMaterial->GetTextureCount(vType); i++)
	{
		aiString Str;
		vMaterial->GetTexture(vType, i, &Str);
		// check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
		bool Skip = false;
		int LoadedTexturesNr = m_TexturesLoaded.size();
		for (unsigned int j = 0; j < LoadedTexturesNr; j++)
		{
			if (std::strcmp(m_TexturesLoaded[j].Path.data(), Str.C_Str()) == 0)
			{
				Textures.push_back(m_TexturesLoaded[j]);
				Skip = true; // a texture with the same filepath has already been loaded, continue to next one. (optimization)
				break;
			}
		}
		if (!Skip)
		{   // if texture hasn't been loaded already, load it
			STexture Texture;
			Texture.ID = __loadTextureFromFile(Str.C_Str(), this->m_Directory);
			Texture.Type = vTypeName;
			Texture.Path = Str.C_Str();
			Textures.push_back(Texture);
			m_TexturesLoaded.push_back(Texture);  // store it as texture loaded for entire model, to ensure we won't unnecesery load duplicate textures.
		}
	}
	return Textures;
}

//****************************************************************************************************
//FUNCTION:
unsigned int CModel::__loadTextureFromFile(const std::string& vPath, const std::string& vDirectory)
{
	std::string Filename = vDirectory + '/' + vPath;

	unsigned int TextureID;
	glGenTextures(1, &TextureID);

	int Width, Height, NrComponents;
	unsigned char *Data = stbi_load(Filename.c_str(), &Width, &Height, &NrComponents, 0);
	if (Data)
	{
		GLenum Format;
		if (NrComponents == 1)
			Format = GL_RED;
		else if (NrComponents == 3)
			Format = GL_RGB;
		else if (NrComponents == 4)
			Format = GL_RGBA;

		glBindTexture(GL_TEXTURE_2D, TextureID);
		glTexImage2D(GL_TEXTURE_2D, 0, Format, Width, Height, 0, Format, GL_UNSIGNED_BYTE, Data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		stbi_image_free(Data);
	}
	else
	{
		throw std::runtime_error("Texture failed to load at path: " + vPath);
		stbi_image_free(Data);
	}

	return TextureID;
}

//****************************************************************************************************
//FUNCTION:
void graphics::CModel::updateVertexPosition(const std::vector<std::vector<glm::vec3>>& vPos, double vAngleThreshold /* = -1.0 */)
{
	const int MeshSize = m_Meshes.size();
	_ASSERTE(MeshSize == vPos.size());

	for (int i = 0; i < MeshSize; i++)
	{
		m_Meshes[i].updateVertexPosition(vPos[i], vAngleThreshold);
	}
}