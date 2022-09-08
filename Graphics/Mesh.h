#pragma once
#include <vector>
#include "Shader.h"

namespace graphics
{
	struct SVertex {
		glm::vec3 Position;
		glm::vec3 Normal;
		glm::vec2 TexCoords;
		glm::vec3 Tangent;
		glm::vec3 Bitangent;

		SVertex() = default;
		SVertex(float vPosX, float vPosY, float vPosZ, float vNormalX, float vNormalY, float vNormalZ, float vTexCoordU, float vTexCoordV)
		{
			Position = glm::vec3(vPosX, vPosY, vPosZ);
			Normal = glm::vec3(vNormalX, vNormalY, vNormalZ);
			TexCoords = glm::vec2(vTexCoordU, vTexCoordV);
		}
	};

	struct STexture {
		unsigned int ID;
		std::string Type;
		std::string Path;
	};

	class CMesh
	{
	public:
		CMesh() = default;
		CMesh(const std::vector<SVertex>& vVertices, const std::vector<unsigned int>& vIndices, const std::vector<STexture>& vTextures);
		CMesh(const CMesh& vMesh);
		CMesh(CMesh&& vMesh) noexcept;
		virtual ~CMesh();

		CMesh& operator=(const CMesh& vMesh);
		CMesh& operator=(CMesh&& vMesh) noexcept;

		void draw(const CShader& vShader);
		void drawInstance(const CShader& vShader, int vInstanceSize);
		void updateVertexPosition(const std::vector<glm::vec3>& vPos, double vAngleThreshold = -1.0);

		unsigned int getVAOID() const;
		int getNumOfIndices() const;
		unsigned int getIndexAt(int vIndex) const;
		int getNumOfVertices() const;
		const SVertex& getVertexAt(int vIndex) const;

		//TODO: ͬһ��ģ�͵�����meshӦ������ͬ��ʵ������Ϣ������ʵ������Ҫͬʱ֧�ֶ�mesh����ʵ�������model����ʵ����
		//���������1������Ѱ��model��mesh���ݵ�ʵ����������2����GraphicsInterface��Ӷ�model��mesh����ʵ�����Ľӿ�(���Ȳ��÷���1)
		unsigned int generateInstanceID() const;
		void setInstanceMatrix(unsigned int vInstanceID, unsigned int vLocationStart, std::vector<glm::mat4> vInstanceMatrix);
		template<class T>
		void setInstanceArray(unsigned int vInstanceID, unsigned int vLocation, const std::vector<T>& vInstanceArray, int vStride);

	protected:
		void _setupMesh();
		void _setupMesh(const std::vector<SVertex>& vVertices, const std::vector<unsigned int>& vIndices);
		void _recalculateNormals(double vAngleThreshold);
		void _clear();

	private:
		void __bindTextures(const CShader& vShader);

		unsigned int m_VAO, m_VBO, m_EBO;

		std::vector<SVertex> m_Vertices;
		std::vector<unsigned int> m_Indices;
		std::vector<STexture> m_Textures;
	};

	//****************************************************************************************************
	//FUNCTION:
	template<class T>
	void graphics::CMesh::setInstanceArray(unsigned int vInstanceID, unsigned int vLocation, const std::vector<T>& vInstanceArray, int vStride)
	{
		glBindVertexArray(m_VAO);
		glBindBuffer(GL_ARRAY_BUFFER, vInstanceID);
		glBufferData(GL_ARRAY_BUFFER, vInstanceArray.size() * sizeof(T), &vInstanceArray[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(vLocation);
		glVertexAttribPointer(vLocation, vStride, GL_FLOAT, GL_FALSE, vStride * sizeof(float), (void*)0);
		glVertexAttribDivisor(vLocation, 1);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
}