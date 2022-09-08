#include "Camera.h"

using namespace graphics;

CCamera::CCamera(const glm::vec3& vPosition /*= glm::vec3(0.0f, 0.0f, 0.0f)*/, const glm::vec3& vUp /*= glm::vec3(0.0f, 1.0f, 0.0f)*/, float vYaw /*= -90.0f*/, float vPitch /*= 0.0f*/, float vMovementSpeed /*= 2.5f*/, float vMouseSensitivity /*= 0.1f*/, float vZoom /*= 45.0f*/)
{
	m_Front = glm::vec3(0.0f, 0.0f, -1.0f);
	m_MovementSpeed = vMovementSpeed;
	m_MouseSensitivity = vMouseSensitivity;
	m_Zoom = vZoom;
	m_Position = vPosition;
	m_WorldUp = vUp;
	m_Yaw = vYaw;
	m_Pitch = vPitch;
	__updateCameraVectors();
}

CCamera::CCamera(float vPosX, float vPosY, float vPosZ, float vUpX, float vUpY, float vUpZ, float vYaw, float vPitch, float vMovementSpeed /*= 2.5f*/, float vMouseSensitivity /*= 0.1f*/, float vZoom /*= 45.0f*/)
{
	m_Front = glm::vec3(0.0f, 0.0f, -1.0f);
	m_MovementSpeed = vMovementSpeed;
	m_MouseSensitivity = vMouseSensitivity;
	m_Zoom = vZoom;
	m_Position = glm::vec3(vPosX, vPosY, vPosZ);
	m_WorldUp = glm::vec3(vUpX, vUpY, vUpZ);
	m_Yaw = vYaw;
	m_Pitch = vPitch;
	__updateCameraVectors();
}

//****************************************************************************************************
//FUNCTION:
void CCamera::processKeyboard(ECameraMovement vDirection, float vDeltaTime)
{
	float Velocity = m_MovementSpeed * vDeltaTime;
	if (vDirection == FORWARD)
		m_Position += m_Front * Velocity;
	if (vDirection == BACKWARD)
		m_Position -= m_Front * Velocity;
	if (vDirection == LEFT)
		m_Position -= m_Right * Velocity;
	if (vDirection == RIGHT)
		m_Position += m_Right * Velocity;
}

//****************************************************************************************************
//FUNCTION:
void CCamera::processMouseMovement(float vXOffset, float vYOffset, GLboolean vConstrainPitch /*= true*/)
{
	vXOffset *= m_MouseSensitivity;
	vYOffset *= m_MouseSensitivity;

	m_Yaw += vXOffset;
	m_Pitch += vYOffset;

	// Make sure that when pitch is out of bounds, screen doesn't get flipped
	if (vConstrainPitch)
	{
		if (m_Pitch > 89.0f)
			m_Pitch = 89.0f;
		if (m_Pitch < -89.0f)
			m_Pitch = -89.0f;
	}

	// Update Front, Right and Up Vectors using the updated Euler angles
	__updateCameraVectors();
}

//****************************************************************************************************
//FUNCTION:
void CCamera::processMouseScroll(float vYOffset)
{
	if (m_Zoom >= 1.0f && m_Zoom <= 45.0f)
		m_Zoom -= vYOffset;
	if (m_Zoom <= 1.0f)
		m_Zoom = 1.0f;
	if (m_Zoom >= 45.0f)
		m_Zoom = 45.0f;
}

//****************************************************************************************************
//FUNCTION:
void CCamera::__updateCameraVectors()
{
	// Calculate the new Front vector
	glm::vec3 Front;
	Front.x = cos(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
	Front.y = sin(glm::radians(m_Pitch));
	Front.z = sin(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
	m_Front = glm::normalize(Front);
	// Also re-calculate the Right and Up vector
	m_Right = glm::normalize(glm::cross(m_Front, m_WorldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
	m_Up = glm::normalize(glm::cross(m_Right, m_Front));
}