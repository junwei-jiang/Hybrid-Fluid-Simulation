#include "Shader.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace graphics;

CShader::CShader(const char * vVertexPath, const char * vFragmentPath, const char * vGeometryPath)
{
	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::string geometryCode;
	std::ifstream vShaderFile;
	std::ifstream fShaderFile;
	std::ifstream gShaderFile;
	// ensure ifstream objects can throw exceptions:
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		// open files
		vShaderFile.open(vVertexPath);
		fShaderFile.open(vFragmentPath);
		std::stringstream vShaderStream, fShaderStream;
		// read file's buffer contents into streams
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		// close file handlers
		vShaderFile.close();
		fShaderFile.close();
		// convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
		// if geometry shader path is present, also load a geometry shader
		if (vGeometryPath != nullptr)
		{
			gShaderFile.open(vGeometryPath);
			std::stringstream gShaderStream;
			gShaderStream << gShaderFile.rdbuf();
			gShaderFile.close();
			geometryCode = gShaderStream.str();
		}
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}
	const char* vShaderCode = vertexCode.c_str();
	const char * fShaderCode = fragmentCode.c_str();
	// 2. compile shaders
	unsigned int vertex, fragment;
	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	__checkCompileErrors(vertex, "VERTEX");
	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	__checkCompileErrors(fragment, "FRAGMENT");
	// if geometry shader is given, compile geometry shader
	unsigned int geometry;
	if (vGeometryPath != nullptr)
	{
		const char * gShaderCode = geometryCode.c_str();
		geometry = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometry, 1, &gShaderCode, NULL);
		glCompileShader(geometry);
		__checkCompileErrors(geometry, "GEOMETRY");
	}
	// shader Program
	m_ID = glCreateProgram();
	glAttachShader(m_ID, vertex);
	glAttachShader(m_ID, fragment);
	if (vGeometryPath != nullptr)
		glAttachShader(m_ID, geometry);
	glLinkProgram(m_ID);
	__checkCompileErrors(m_ID, "PROGRAM");
	// delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);
	if (vGeometryPath != nullptr)
		glDeleteShader(geometry);
}

//****************************************************************************************************
//FUNCTION:
void CShader::use()
{
	glUseProgram(m_ID);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setBool(const std::string & vName, bool vValue) const
{
	glUniform1i(glGetUniformLocation(m_ID, vName.c_str()), (int)vValue);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setInt(const std::string & vName, int vValue) const
{
	glUniform1i(glGetUniformLocation(m_ID, vName.c_str()), vValue);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setFloat(const std::string & vName, float vValue) const
{
	glUniform1f(glGetUniformLocation(m_ID, vName.c_str()), vValue);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setVec2(const std::string & vName, const glm::vec2 & vValue) const
{
	glUniform2fv(glGetUniformLocation(m_ID, vName.c_str()), 1, &vValue[0]);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setVec2(const std::string & vName, float vX, float vY) const
{
	glUniform2f(glGetUniformLocation(m_ID, vName.c_str()), vX, vY);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setVec3(const std::string & vName, const glm::vec3 & vValue) const
{
	glUniform3fv(glGetUniformLocation(m_ID, vName.c_str()), 1, &vValue[0]);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setVec3(const std::string & vName, float vX, float vY, float vZ) const
{
	glUniform3f(glGetUniformLocation(m_ID, vName.c_str()), vX, vY, vZ);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setVec4(const std::string & vName, const glm::vec4 & vValue) const
{
	glUniform4fv(glGetUniformLocation(m_ID, vName.c_str()), 1, &vValue[0]);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setVec4(const std::string & vName, float vX, float vY, float vZ, float vW)
{
	glUniform4f(glGetUniformLocation(m_ID, vName.c_str()), vX, vY, vZ, vW);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setMat2(const std::string & vName, const glm::mat2 & vMat) const
{
	glUniformMatrix2fv(glGetUniformLocation(m_ID, vName.c_str()), 1, GL_FALSE, &vMat[0][0]);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setMat3(const std::string & vName, const glm::mat3 & vMat) const
{
	glUniformMatrix3fv(glGetUniformLocation(m_ID, vName.c_str()), 1, GL_FALSE, &vMat[0][0]);
}

//****************************************************************************************************
//FUNCTION:
void CShader::setMat4(const std::string & vName, const glm::mat4 & vMat) const
{
	glUniformMatrix4fv(glGetUniformLocation(m_ID, vName.c_str()), 1, GL_FALSE, &vMat[0][0]);
}

//****************************************************************************************************
//FUNCTION:
void CShader::__checkCompileErrors(GLuint vShader, std::string vType)
{
	GLint success;
	GLchar infoLog[1024];
	if (vType != "PROGRAM")
	{
		glGetShaderiv(vShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vShader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << vType << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(vShader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(vShader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << vType << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}