#include "Utils.h"

graphics::CCamera *graphics::g_pCamera = nullptr;
float graphics::g_LastX = 0.0f;
float graphics::g_LastY = 0.0f;
bool graphics::g_FirstMouse = true;
bool graphics::g_Keys[1024] = { false };
bool graphics::g_KeysPressed[1024] = { false };

//****************************************************************************************************
//FUNCTION:
void graphics::processInput(GLFWwindow * vWindow, float vDeltaTime)
{
	if (nullptr != g_pCamera)
	{
		if (glfwGetKey(vWindow, GLFW_KEY_W) == GLFW_PRESS)
			g_pCamera->processKeyboard(FORWARD, vDeltaTime);
		if (glfwGetKey(vWindow, GLFW_KEY_S) == GLFW_PRESS)
			g_pCamera->processKeyboard(BACKWARD, vDeltaTime);
		if (glfwGetKey(vWindow, GLFW_KEY_A) == GLFW_PRESS)
			g_pCamera->processKeyboard(LEFT, vDeltaTime);
		if (glfwGetKey(vWindow, GLFW_KEY_D) == GLFW_PRESS)
			g_pCamera->processKeyboard(RIGHT, vDeltaTime);
	}
}

//****************************************************************************************************
//FUNCTION:
void graphics::keyCallback(GLFWwindow* vWindow, int vKey, int vScancode, int vAction, int vMode)
{
	if (vKey == GLFW_KEY_ESCAPE && vAction == GLFW_PRESS)
		glfwSetWindowShouldClose(vWindow, GL_TRUE);

	if (vKey >= 0 && vKey <= 1024)
	{
		if (vAction == GLFW_PRESS)
			g_Keys[vKey] = true;
		else if (vAction == GLFW_RELEASE)
		{
			g_Keys[vKey] = false;
			g_KeysPressed[vKey] = false;
		}
	}
}

//****************************************************************************************************
//FUNCTION:
void graphics::mouseCallback(GLFWwindow* vWindow, double vXPos, double vYPos)
{
	if (g_FirstMouse)
	{
		g_LastX = vXPos;
		g_LastY = vYPos;
		g_FirstMouse = false;
	}

	float XOffset = vXPos - g_LastX;
	float YOffset = g_LastY - vYPos; // reversed since y-coordinates go from bottom to top

	g_LastX = vXPos;
	g_LastY = vYPos;

	if(nullptr != g_pCamera)
		g_pCamera->processMouseMovement(XOffset, YOffset);
}

//****************************************************************************************************
//FUNCTION:
void graphics::scrollCallback(GLFWwindow* vWindow, double vXOffset, double vYOffset)
{
	if (nullptr != g_pCamera)
		g_pCamera->processMouseScroll(vYOffset);
}