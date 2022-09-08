#pragma once
#include "glad.h"
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace graphics
{
	// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
	enum ECameraMovement {
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT
	};

	// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
	class CCamera
	{
	public:
		// Constructor with vectors
		CCamera(const glm::vec3& vPosition = glm::vec3(0.0f, 0.0f, 0.0f), const glm::vec3& vUp = glm::vec3(0.0f, 1.0f, 0.0f), float vYaw = -90.0f, float vPitch = 0.0f, float vMovementSpeed = 0.01f, float vMouseSensitivity = 0.01f, float vZoom = 45.0f);
		// Constructor with scalar values
		CCamera(float vPosX, float vPosY, float vPosZ, float vUpX, float vUpY, float vUpZ, float vYaw, float vPitch, float vMovementSpeed = 0.01f, float vMouseSensitivity = 0.01f, float vZoom = 45.0f);

		// Get Method
		glm::mat4 getViewMatrix() const { return glm::lookAt(m_Position, m_Position + m_Front, m_Up); }
		glm::vec3 getLookAt() const { return m_Position + m_Front; }
		const glm::vec3& getPosition() const { return m_Position; }
		const glm::vec3& getFront() const { return m_Front; }
		const glm::vec3& getUp() const { return m_Up; }
		float getYaw() const { return m_Yaw; }
		float getPitch() const { return m_Pitch; }
		float getMovementSpeed() const { return m_MovementSpeed; }
		float getMouseSensitivity() const { return m_MouseSensitivity; }
		float getZoom() const { return m_Zoom; }

		// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
		void processKeyboard(ECameraMovement vDirection, float vDeltaTime);
		// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
		void processMouseMovement(float vXOffset, float vYOffset, GLboolean vConstrainPitch = true);
		// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
		void processMouseScroll(float vYOffset);

	private:
		// Calculates the front vector from the Camera's (updated) Euler Angles
		void __updateCameraVectors();

		// Camera Attributes
		glm::vec3 m_Position;
		glm::vec3 m_Front;
		glm::vec3 m_Up;
		glm::vec3 m_Right;
		glm::vec3 m_WorldUp;
		// Euler Angles
		float m_Yaw;
		float m_Pitch;
		// Camera options
		float m_MovementSpeed;
		float m_MouseSensitivity;
		float m_Zoom;
	};
}