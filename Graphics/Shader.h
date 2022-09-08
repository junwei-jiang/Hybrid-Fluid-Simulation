#pragma once
#include "glad.h"
#include <glm/glm.hpp>
#include <string>

namespace graphics
{
	class CShader
	{
	public:
		// constructor generates the shader on the fly
		CShader(const char* vVertexPath, const char* vFragmentPath, const char* vGeometryPath = nullptr);

		// activate the shader
		void use();

		unsigned int getID() const { return m_ID; }

		// utility uniform functions
		void setBool(const std::string &vName, bool vValue) const;
		void setInt(const std::string &vName, int vValuevValue) const;
		void setFloat(const std::string &vName, float vValue) const;
		void setVec2(const std::string &vName, const glm::vec2 &vValue) const;
		void setVec2(const std::string &vName, float vX, float vY) const;
		void setVec3(const std::string &vName, const glm::vec3 &vValue) const;
		void setVec3(const std::string &vName, float vX, float vY, float vZ) const;
		void setVec4(const std::string &vName, const glm::vec4 &vValue) const;
		void setVec4(const std::string &vName, float vX, float vY, float vZ, float vW);
		void setMat2(const std::string &vName, const glm::mat2 &vMat) const;
		void setMat3(const std::string &vName, const glm::mat3 &vMat) const;
		void setMat4(const std::string &vName, const glm::mat4 &vMat) const;

	private:
		// utility function for checking shader compilation/linking errors.
		void __checkCompileErrors(GLuint vShader, std::string vType);

		unsigned int m_ID;
	};
}