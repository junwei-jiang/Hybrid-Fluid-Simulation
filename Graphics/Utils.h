#pragma once
#include "glad.h"
#include "glfw/glfw3.h"
#include "Camera.h"
#include <functional>

namespace graphics
{
	void processInput(GLFWwindow* vWindow, float vDeltaTime);
	void keyCallback(GLFWwindow* vWindow, int vKey, int vScancode, int vAction, int vMode);
	void mouseCallback(GLFWwindow* vWindow, double vXPos, double vYPos);
	void scrollCallback(GLFWwindow* vWindow, double vXOffset, double vYOffset);

	extern graphics::CCamera *g_pCamera;
	extern float g_LastX;
	extern float g_LastY;
	extern bool g_FirstMouse;
	extern bool g_Keys[1024];
	extern bool g_KeysPressed[1024];
}