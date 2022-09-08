#pragma once
#include "Utils.h"
#include "stb_image.h"
#include <functional>

//TODO: 动态库导出接口
namespace graphics
{
	void createWindow(GLFWwindow*& voWindow, int vMajor, int vMinor, int vWidth, int vHeight);
	void closeWindow(GLFWwindow* voWindow);

	void createCamera(GLFWwindow* vWindow, graphics::CCamera* vioCamera);

	void mainRunLoop(GLFWwindow* vWindow, const std::function<void()>& vRunLoop);

	void renderQuad();

	void captureFrameBuffer(const std::string & vPath, int vWidth, int vHeight, GLuint* voTempBufferIndex);

	unsigned int loadTexture(const std::string& vPath);
	unsigned int loadCubemap(const std::vector<std::string>& vFacesPath);
}