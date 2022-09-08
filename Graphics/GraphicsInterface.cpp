#include "GraphicsInterface.h"
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//****************************************************************************************************
//FUNCTION:
void graphics::createWindow(GLFWwindow*& voWindow, int vMajor, int vMinor, int vWidth, int vHeight)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, vMajor);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, vMinor);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	voWindow = glfwCreateWindow(vWidth, vHeight, "Simulation", NULL, NULL);
	if (voWindow == NULL)
	{
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(voWindow);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		return ;

	glViewport(0, 0, vWidth, vHeight);

	glfwSetFramebufferSizeCallback(voWindow, [](GLFWwindow* window, int width, int height) {
		glViewport(0, 0, width, height);
	});

	g_LastX = vWidth / 2.0f;
	g_LastY = vHeight / 2.0f;
}

//****************************************************************************************************
//FUNCTION:
void graphics::closeWindow(GLFWwindow* voWindow)
{
	g_pCamera = nullptr;
	voWindow = nullptr;
	glfwTerminate();
}

//****************************************************************************************************
//FUNCTION:
void graphics::createCamera(GLFWwindow* vWindow, CCamera* vioCamera)
{
	if (vioCamera != nullptr)
		g_pCamera = vioCamera;
	else
		g_pCamera = new CCamera();

	glfwSetKeyCallback(vWindow, keyCallback);
	glfwSetCursorPosCallback(vWindow, mouseCallback);
	glfwSetScrollCallback(vWindow, scrollCallback);

	glfwSetInputMode(vWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

//****************************************************************************************************
//FUNCTION:
void graphics::mainRunLoop(GLFWwindow* vWindow, const std::function<void()>& vRunLoop)
{
	float LastFrame = 0.0f;
	float DeltaTime = 0.0f;

	while (!glfwWindowShouldClose(vWindow))
	{
		float CurrentFrame = glfwGetTime();
		DeltaTime = CurrentFrame - LastFrame;
		LastFrame = CurrentFrame;

		processInput(vWindow, DeltaTime);

		vRunLoop();

		glfwPollEvents();
		glfwSwapBuffers(vWindow);
	}
}

//****************************************************************************************************
//FUNCTION:
void graphics::renderQuad()
{
	static unsigned int QuadVAO = 0;
	static unsigned int QuadVBO = 0;

	if (QuadVAO == 0)
	{
		float QuadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &QuadVAO);
		glGenBuffers(1, &QuadVBO);
		glBindVertexArray(QuadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, QuadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(QuadVertices), &QuadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}
	glBindVertexArray(QuadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

//****************************************************************************************************
//FUNCTION:
unsigned int graphics::loadTexture(const std::string& vPath)
{
	unsigned int TextureID;
	glGenTextures(1, &TextureID);

	int Width, Height, NrComponents;
	unsigned char *pData = stbi_load(vPath.c_str(), &Width, &Height, &NrComponents, 0);
	if (pData)
	{
		GLenum Format;
		if (NrComponents == 1)
			Format = GL_RED;
		else if (NrComponents == 3)
			Format = GL_RGB;
		else if (NrComponents == 4)
			Format = GL_RGBA;

		glBindTexture(GL_TEXTURE_2D, TextureID);
		glTexImage2D(GL_TEXTURE_2D, 0, Format, Width, Height, 0, Format, GL_UNSIGNED_BYTE, pData);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		stbi_image_free(pData);
	}
	else
	{
		throw std::ios_base::failure("Texture failed to load at path: " + vPath);
		stbi_image_free(pData);
	}

	return TextureID;
}

//****************************************************************************************************
//FUNCTION:
unsigned int graphics::loadCubemap(const std::vector<std::string>& vFacesPath)
{
	unsigned int TextureID;
	glGenTextures(1, &TextureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, TextureID);

	int Width, Height, NrChannels;
	for (unsigned int i = 0; i < vFacesPath.size(); i++)
	{
		unsigned char *pData = stbi_load(vFacesPath[i].c_str(), &Width, &Height, &NrChannels, 0);
		if (pData)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
				0, GL_RGB, Width, Height, 0, GL_RGB, GL_UNSIGNED_BYTE, pData
			);
			stbi_image_free(pData);
		}
		else
		{
			throw std::ios_base::failure("Cubemap texture failed to load at path: " + vFacesPath[i]);
			stbi_image_free(pData);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return TextureID;
}

//****************************************************************************************************
//FUNCTION:
void graphics::captureFrameBuffer(const std::string & vPath, int vWidth, int vHeight, GLuint* voTempBufferIndex)
{
	glBindBuffer(GL_ARRAY_BUFFER, *voTempBufferIndex);
	glBufferData(GL_ARRAY_BUFFER, sizeof(int) * vWidth * vHeight * 3, NULL, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, *voTempBufferIndex);
	glReadPixels(0, 0, vWidth, vHeight, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	int *data = (int *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	stbi_write_bmp(vPath.c_str(), vWidth, vHeight, 3, data);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
}