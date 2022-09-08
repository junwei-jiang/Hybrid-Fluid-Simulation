#include "Mesh.h"
#include <unordered_map>

using namespace graphics;

CMesh::CMesh(const std::vector<SVertex>& vVertices, const std::vector<unsigned int>& vIndices, const std::vector<STexture>& vTextures)
{
	this->m_Vertices = vVertices;
	this->m_Indices = vIndices;
	this->m_Textures = vTextures;

	_setupMesh();
}

graphics::CMesh::CMesh(const CMesh& vMesh)
	: m_Vertices(vMesh.m_Vertices), m_Indices(vMesh.m_Indices), m_Textures(vMesh.m_Textures),
	m_VAO(vMesh.m_VAO), m_VBO(vMesh.m_VBO), m_EBO(vMesh.m_EBO) {}

graphics::CMesh::CMesh(CMesh&& vMesh) noexcept
	: m_Vertices(std::move(vMesh.m_Vertices)), m_Indices(std::move(vMesh.m_Indices)), m_Textures(std::move(vMesh.m_Textures)),
	m_VAO(vMesh.m_VAO), m_VBO(vMesh.m_VBO), m_EBO(vMesh.m_EBO)
{
	vMesh.m_VAO = 0;
	vMesh.m_VBO = 0;
	vMesh.m_EBO = 0;
}

graphics::CMesh::~CMesh()
{
	_clear();
}

//****************************************************************************************************
//FUNCTION:
graphics::CMesh& graphics::CMesh::operator=(CMesh&& vMesh) noexcept
{
	if (this != &vMesh)
	{
		this->m_Vertices = vMesh.m_Vertices;
		this->m_Indices = vMesh.m_Indices;
		this->m_Textures = vMesh.m_Textures;
		this->m_VAO = vMesh.m_VAO;
		this->m_VBO = vMesh.m_VBO;
		this->m_EBO = vMesh.m_EBO;

		vMesh.m_VAO = 0;
		vMesh.m_VBO = 0;
		vMesh.m_EBO = 0;
	}

	return *this;
}

//****************************************************************************************************
//FUNCTION:
graphics::CMesh& graphics::CMesh::operator=(const CMesh& vMesh)
{
	if (this != &vMesh)
	{
		this->m_Vertices = vMesh.m_Vertices;
		this->m_Indices = vMesh.m_Indices;
		this->m_Textures = vMesh.m_Textures;
		this->m_VAO = vMesh.m_VAO;
		this->m_VBO = vMesh.m_VBO;
		this->m_EBO = vMesh.m_EBO;
	}

	return *this;
}

//****************************************************************************************************
//FUNCTION:
void CMesh::draw(const CShader& vShader)
{
	if (!m_Textures.empty())
	{
		__bindTextures(vShader);
	}

	// draw mesh
	glBindVertexArray(m_VAO);
	glDrawElements(GL_TRIANGLES, m_Indices.size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	// always good practice to set everything back to defaults once configured.
	glActiveTexture(GL_TEXTURE0);
}

//****************************************************************************************************
//FUNCTION:
void graphics::CMesh::drawInstance(const CShader& vShader, int vInstanceSize)
{
	if (!m_Textures.empty())
	{
		__bindTextures(vShader);
	}

	// draw mesh
	glBindVertexArray(m_VAO);
	glDrawElementsInstanced(GL_TRIANGLES, m_Indices.size(), GL_UNSIGNED_INT, 0, vInstanceSize);
	glBindVertexArray(0);

	// always good practice to set everything back to defaults once configured.
	glActiveTexture(GL_TEXTURE0);
}

//****************************************************************************************************
//FUNCTION:
void graphics::CMesh::updateVertexPosition(const std::vector<glm::vec3>& vPos, double vAngleThreshold /* = -1.0 */)
{
	const int VerticesSize = m_Vertices.size();
	_ASSERTE(VerticesSize == vPos.size());
	for (int i = 0; i < VerticesSize; i++)
	{
		m_Vertices[i].Position = vPos[i];
		m_Vertices[i].Normal = glm::vec3(0.0f);
	}

	if (vAngleThreshold > 0)
	{
		_recalculateNormals(vAngleThreshold);
	}
	else
	{
		_ASSERTE(m_Indices.size() % 3 == 0);
		const int IndicesSize = m_Indices.size();
		for (int i = 0; i < IndicesSize; i += 3)
		{
			SVertex& Vertex0 = m_Vertices[m_Indices[i]];
			SVertex& Vertex1 = m_Vertices[m_Indices[i + 1]];
			SVertex& Vertex2 = m_Vertices[m_Indices[i + 2]];

			glm::vec3 Vector1 = Vertex1.Position - Vertex0.Position;
			glm::vec3 Vector2 = Vertex2.Position - Vertex0.Position;

			glm::vec3 Normal = glm::normalize(glm::cross(Vector1, Vector2));

			Vertex0.Normal += Normal;
			Vertex1.Normal += Normal;
			Vertex2.Normal += Normal;
		}
		for (int i = 0; i < VerticesSize; i++)
		{
			m_Vertices[i].Normal = glm::normalize(m_Vertices[i].Normal);
		}
	}

	glDeleteVertexArrays(1, &m_VAO);
	glDeleteBuffers(1, &m_VBO);
	glDeleteBuffers(1, &m_EBO);
	_setupMesh();
}

//****************************************************************************************************
//FUNCTION:
unsigned int graphics::CMesh::getIndexAt(int vIndex) const
{
	_ASSERTE(vIndex < m_Indices.size());
	return m_Indices[vIndex];
}

//****************************************************************************************************
//FUNCTION:
int graphics::CMesh::getNumOfVertices() const
{
	return m_Vertices.size();
}

//****************************************************************************************************
//FUNCTION:
const graphics::SVertex& graphics::CMesh::getVertexAt(int vIndex) const
{
	_ASSERTE(vIndex >= 0 && vIndex < m_Vertices.size());
	return m_Vertices[vIndex];
}

//****************************************************************************************************
//FUNCTION:
unsigned int graphics::CMesh::generateInstanceID() const
{
	unsigned int InstanceVBO;
	glGenBuffers(1, &InstanceVBO);
	return InstanceVBO;
}

//****************************************************************************************************
//FUNCTION:
void graphics::CMesh::setInstanceMatrix(unsigned int vInstanceID, unsigned int vLocationStart, std::vector<glm::mat4> vInstanceMatrix)
{
	glBindVertexArray(m_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, vInstanceID);
	glBufferData(GL_ARRAY_BUFFER, vInstanceMatrix.size() * sizeof(glm::mat4), &vInstanceMatrix[0], GL_STATIC_DRAW);

	// vertex property.
	GLsizei Vec4Size = sizeof(glm::vec4);
	for (int i = 0; i < 4; i++)
	{
		GLuint Index = vLocationStart + i;
		glEnableVertexAttribArray(Index);
		glVertexAttribPointer(Index, 4, GL_FLOAT, GL_FALSE, 4 * Vec4Size, (void*)(i*Vec4Size));
		glVertexAttribDivisor(Index, 1);
	}

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//****************************************************************************************************
//FUNCTION:
unsigned int CMesh::getVAOID() const
{
	_ASSERTE(m_VAO > 0);
	return m_VAO;
}

//****************************************************************************************************
//FUNCTION:
int CMesh::getNumOfIndices() const
{
	_ASSERTE(m_Indices.size() > 0);
	return m_Indices.size();
}

//****************************************************************************************************
//FUNCTION:
void CMesh::_setupMesh()
{
	// create buffers/arrays
	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_VBO);
	glGenBuffers(1, &m_EBO);

	glBindVertexArray(m_VAO);
	// load data into vertex buffers
	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	// A great thing about struct is that their memory layout is sequential for all its items.
	// The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
	// again translates to 3/2 floats which translates to a byte array.
	glBufferData(GL_ARRAY_BUFFER, m_Vertices.size() * sizeof(SVertex), &m_Vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Indices.size() * sizeof(unsigned int), &m_Indices[0], GL_STATIC_DRAW);

	// set the vertex attribute pointers
	// vertex Positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SVertex), (void*)0);
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SVertex), (void*)offsetof(SVertex, Normal));
	// vertex texture coordinates
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(SVertex), (void*)offsetof(SVertex, TexCoords));
	// vertex tangent
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(SVertex), (void*)offsetof(SVertex, Tangent));
	// vertex bi tangent
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(SVertex), (void*)offsetof(SVertex, Bitangent));

	glBindVertexArray(0);
}

//****************************************************************************************************
//FUNCTION:
void graphics::CMesh::_setupMesh(const std::vector<SVertex>& vVertices, const std::vector<unsigned int>& vIndices)
{
	this->m_Vertices = vVertices;
	this->m_Indices = vIndices;

	_setupMesh();
}

struct HashFunc
{
	std::size_t operator()(const glm::vec3& key) const
	{
		const long X = static_cast<long>(glm::round(key.x * Tolerance));
		const long Y = static_cast<long>(glm::round(key.x * Tolerance));
		const long Z = static_cast<long>(glm::round(key.x * Tolerance));

		long Result = FNV32Init;
		Result ^= X;
		Result *= FNV32Prime;
		Result ^= Y;
		Result *= FNV32Prime;
		Result ^= Z;
		Result *= FNV32Prime;

		return std::hash<long>()(Result);
	}

private:
	// Change this if you require a different precision.
	const int Tolerance = 100000;

	// Magic FNV values. Do not change these.
	const long FNV32Init = 0x811c9dc5;
	const long FNV32Prime = 0x01000193;
};

struct EqualKey
{
	bool operator()(const glm::vec3& left, const glm::vec3& right) const
	{
		const long LeftX = static_cast<long>(glm::round(left.x * Tolerance));
		const long LeftY = static_cast<long>(glm::round(left.x * Tolerance));
		const long LeftZ = static_cast<long>(glm::round(left.x * Tolerance));

		const long RightX = static_cast<long>(glm::round(right.x * Tolerance));
		const long RightY = static_cast<long>(glm::round(right.x * Tolerance));
		const long RightZ = static_cast<long>(glm::round(right.x * Tolerance));

		return LeftX == RightX && LeftY == RightY && LeftZ == RightZ;
	}

private:
	// Change this if you require a different precision.
	const int Tolerance = 100000;
};

//****************************************************************************************************
//FUNCTION:
void graphics::CMesh::_recalculateNormals(double vAngleThreshold)
{
	const int IndicesSize = m_Indices.size();
	_ASSERTE(IndicesSize % 3 == 0);
	std::vector<glm::vec3> TriangleNormals(IndicesSize / 3);

	typedef std::vector<std::pair<int, int>> VertexEntryArray; // first: Triangle Index, second: Vertex Index
	std::unordered_map<glm::vec3, VertexEntryArray, HashFunc, EqualKey> Dictionary;

	for (int i = 0; i < IndicesSize; i += 3)
	{
		SVertex& Vertex0 = m_Vertices[m_Indices[i]];
		SVertex& Vertex1 = m_Vertices[m_Indices[i + 1]];
		SVertex& Vertex2 = m_Vertices[m_Indices[i + 2]];

		glm::vec3 Vector1 = Vertex1.Position - Vertex0.Position;
		glm::vec3 Vector2 = Vertex2.Position - Vertex0.Position;

		glm::vec3 Normal = glm::normalize(glm::cross(Vector1, Vector2));
		int TriangleIndex = i / 3;
		TriangleNormals[TriangleIndex] = Normal;

		VertexEntryArray& EntryArray0 = Dictionary[Vertex0.Position];
		EntryArray0.push_back({ TriangleIndex, m_Indices[i] });
		VertexEntryArray& EntryArray1 = Dictionary[Vertex1.Position];
		EntryArray1.push_back({ TriangleIndex, m_Indices[i + 1] });
		VertexEntryArray& EntryArray2 = Dictionary[Vertex2.Position];
		EntryArray2.push_back({ TriangleIndex, m_Indices[i + 2] });
	}

	const double CosineThreshold = glm::radians(vAngleThreshold);
	for (auto Iter = Dictionary.begin(); Iter != Dictionary.end(); Iter++)
	{
		VertexEntryArray VertexEntries = Iter->second;
		for (auto LeftEntry : VertexEntries)
		{
			glm::vec3 SumNormal = glm::vec3(0);
			for (auto RightEntry : VertexEntries)
			{
				if (LeftEntry.second == RightEntry.second)
				{
					SumNormal += TriangleNormals[RightEntry.first];
				}
				else
				{
					const double DotRes = glm::dot(TriangleNormals[LeftEntry.first], TriangleNormals[RightEntry.first]);
					if (DotRes >= CosineThreshold)
					{
						SumNormal += TriangleNormals[RightEntry.first];
					}
				}
			}

			m_Vertices[LeftEntry.second].Normal = glm::normalize(SumNormal);
		}
	}
}

//****************************************************************************************************
//FUNCTION:
void graphics::CMesh::_clear()
{
	std::vector<SVertex>().swap(m_Vertices);
	std::vector<unsigned int>().swap(m_Indices);
	glDeleteVertexArrays(1, &m_VAO);
	glDeleteBuffers(1, &m_VBO);
	glDeleteBuffers(1, &m_EBO);
}

//****************************************************************************************************
//FUNCTION:
void graphics::CMesh::__bindTextures(const CShader& vShader)
{
	// bind appropriate textures
	unsigned int DiffuseNr = 1;
	unsigned int SpecularNr = 1;
	unsigned int NormalNr = 1;
	unsigned int HeightNr = 1;
	unsigned int ReflectionNr = 1;
	for (unsigned int i = 0; i < m_Textures.size(); i++)
	{
		glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
		// retrieve texture number (the N in diffuse_textureN)
		std::string Number;
		std::string Name = m_Textures[i].Type;
		if (Name == "texture_diffuse")
			Number = std::to_string(DiffuseNr++);
		else if (Name == "texture_specular")
			Number = std::to_string(SpecularNr++); // transfer unsigned int to stream
		else if (Name == "texture_normal")
			Number = std::to_string(NormalNr++); // transfer unsigned int to stream
		else if (Name == "texture_height")
			Number = std::to_string(HeightNr++); // transfer unsigned int to stream
		else if (Name == "texture_reflection")
			Number = std::to_string(ReflectionNr++); // transfer unsigned int to stream

		// now set the sampler to the correct texture unit
		glUniform1i(glGetUniformLocation(vShader.getID(), (Name + Number).c_str()), i);
		// and finally bind the texture
		glBindTexture(GL_TEXTURE_2D, m_Textures[i].ID);
	}
}
