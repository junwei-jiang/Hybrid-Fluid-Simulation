#include <Shader.h>
#include <Sphere.h>
#include <Plane.h>
#include <Model.h>
#include <GraphicsInterface.h>
#include <memory>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include "../MPhysics/MPhysics.h"

typedef float Real;
#define REAL_MAX FLT_MAX

const int ScreenWidth = 1080;
const int ScreenHeight = 720;
const Real NearPlane = 0.1;
const Real FarPlane = 100;

Real ParticleRadius = 0.025;
GLuint CaptureTempBuffer;

Real ColorMinScalar = 0.0;
Real ColorMaxScalar = 2.0;
bool captureFrame = true;
glm::vec3 InitTransform = glm::vec3(0, 0, 0);
glm::mat4 InitRotation = glm::mat4(1.0);
Real Speed = 0.1;

GLFWwindow* initWindow();
graphics::CCamera* initCamera(GLFWwindow* vWindow);
Real fetchVelRange(const std::vector<Real>& vInput);
void drawParticles
(
	std::vector<Real> vParticlePos,
	std::vector<Real> vParticleVel,
	const glm::mat4& vModel,
	const glm::mat4& vView,
	const glm::mat4& vProjection,
	unsigned int vPFParticleSize
);
void processInput(GLFWwindow* vWindow);
int main()
{
	try
	{
		MPhysicsAPI::initMPhysics();
		MPhysicsAPI::loadScene("./TestScene.json");

		GLFWwindow *Window = initWindow();
		std::shared_ptr<graphics::CModel> MapModel(new graphics::CModel("./Map.obj"));
		std::shared_ptr<graphics::CShader> ShaderForModel = std::shared_ptr<graphics::CShader>(new graphics::CShader("./Model.vs", "./Model.fs"));

		std::shared_ptr<graphics::CCamera> Camera(initCamera(Window));
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glGenBuffers(1, &CaptureTempBuffer);
		int currentFrameIndex = 0;

		MPhysicsAPI::instanceRigidBoundary("TestSimulator", 0);

		graphics::mainRunLoop(Window, [&]()
		{
			processInput(Window);
			glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glm::mat4 Model = glm::mat4(1.0f);
			Model = glm::scale(Model, glm::vec3(0.1f));
			glm::mat4 View = Camera->getViewMatrix();
			glm::mat4 Projection = glm::perspective(Camera->getZoom(), (float)ScreenWidth / (float)ScreenHeight, (float)NearPlane, (float)FarPlane);

			MPhysicsAPI::update(0.016f);

			MPhysicsAPI::transformRigidBoundary
			(
				"TestSimulator",
				0,
				0,
				InitTransform.x, InitTransform.y, InitTransform.z
			);

			unsigned int PFParticleSize = MPhysicsAPI::getParticleSize("TestSimulator");
			unsigned int EulerParticleSize = MPhysicsAPI::getEulerParticleSize("TestSimulator");
			unsigned int ParticleSize = PFParticleSize + EulerParticleSize;
			std::vector<Real> ParticlesPos(ParticleSize * 3);
			std::vector<Real> ParticlesVel(ParticleSize * 3);
			MPhysicsAPI::copyPositionToCPU("TestSimulator", ParticlesPos.data());
			MPhysicsAPI::copyVelToCPU("TestSimulator", ParticlesVel.data());
			MPhysicsAPI::copyEulerParticlesPosToCPU("TestSimulator", ParticlesPos.data() + PFParticleSize * 3);
			MPhysicsAPI::copyEulerParticlesVelToCPU("TestSimulator", ParticlesVel.data() + PFParticleSize * 3);
			drawParticles(ParticlesPos, ParticlesVel, Model, View, Projection, PFParticleSize);

			glCullFace(GL_FRONT);
			ShaderForModel->use();
			ShaderForModel->setMat4("model", Model);
			ShaderForModel->setMat4("view", Camera->getViewMatrix());
			ShaderForModel->setMat4("projection", Projection);
			ShaderForModel->setVec3("ViewPos", Camera->getPosition());
			ShaderForModel->setVec3("LightDir", glm::vec3(0, 0, -1.0f));
			MapModel->draw(*ShaderForModel);
			glCullFace(GL_BACK);

			if (captureFrame)
				graphics::captureFrameBuffer
				(
					std::string("./output/") + std::to_string(currentFrameIndex) + std::string(".bmp"),
					ScreenWidth,
					ScreenHeight,
					&CaptureTempBuffer
				);
			currentFrameIndex++;
		});

		graphics::closeWindow(Window);

		MPhysicsAPI::freeMPhysics();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		system("pause");
	}
}

GLFWwindow* initWindow()
{
	const int GLCoreMajor = 3;
	const int GLCoreMinor = 3;

	GLFWwindow *Window = nullptr;
	graphics::createWindow(Window, GLCoreMajor, GLCoreMinor, ScreenWidth, ScreenHeight);

	return Window;
}

graphics::CCamera* initCamera(GLFWwindow* vWindow)
{
	Real Pos[3] = { -1.0, 0.5, 0.5};
	graphics::CCamera *Camera = new graphics::CCamera({ Pos[0], Pos[1], Pos[2] }, { 0.0, 1.0, 0.0 }, -90.0, 0.0, 0.1, 0.01);
	graphics::createCamera(vWindow, Camera);

	return Camera;
}

Real fetchVelRange(const std::vector<Real>& vInput)
{
	Real MaxVelLength = -REAL_MAX;
	for (int i = 0; i < vInput.size() / 3; i++)
	{
		glm::vec3 Vel = glm::vec3(
			vInput[i * 3],
			vInput[i * 3 + 1],
			vInput[i * 3 + 2]
		);
		Real VelLength = glm::length(Vel);
		if (VelLength > MaxVelLength) MaxVelLength = VelLength;
	}
	return MaxVelLength;
}

void drawParticles
(
	std::vector<Real> vParticlePos,
	std::vector<Real> vParticleVel,
	const glm::mat4& vModel, 
	const glm::mat4& vView, 
	const glm::mat4& vProjection,
	unsigned int vPFParticleSize
)
{
	static bool IsInitialized = false;
	static int ParticleCount = 0;
	static GLuint OffsetInstanceID = 0;
	static GLuint ColorInstanceID = 0;
	static std::shared_ptr<graphics::CShader> ShaderForParticle = nullptr;
	static std::shared_ptr<graphics::CSphere> Sphere = nullptr;
	if (!IsInitialized)
	{
		const std::string ParticleVertex = "./Particle.vs";
		const std::string ParticleFragment = "./Particle.fs";
		ShaderForParticle = std::shared_ptr<graphics::CShader>(new graphics::CShader(ParticleVertex.c_str(), ParticleFragment.c_str()));

		Sphere = std::shared_ptr<graphics::CSphere>(new graphics::CSphere(ParticleRadius * 0.6));

		ParticleCount = vParticlePos.size() / 3;
		std::vector<glm::vec3> Translations(ParticleCount);
		std::vector<glm::vec3> Colors(ParticleCount);
		Real ColorB = 0.0;
		for (int i = 0; i < ParticleCount; i++)
		{
			Translations[i] = glm::vec3(vParticlePos[i * 3], vParticlePos[i * 3 + 1], vParticlePos[i * 3 + 2]);
			Real VelLength = glm::length(glm::vec3(vParticleVel[i * 3], vParticleVel[i * 3 + 1], vParticleVel[i * 3 + 2]));
			Real ClampVel = std::clamp<Real>(VelLength, ColorMinScalar, ColorMaxScalar);
			
			if (i < vPFParticleSize) ColorB = 1.0;
			else ColorB = 1.0;
			Colors[i] = glm::vec3(
				ClampVel / (ColorMaxScalar - ColorMinScalar),
				ClampVel / (ColorMaxScalar - ColorMinScalar),
				ColorB
			);
		}
		OffsetInstanceID = Sphere->generateInstanceID();
		Sphere->setInstanceArray(OffsetInstanceID, 1, Translations, 3);
		ColorInstanceID = Sphere->generateInstanceID();
		Sphere->setInstanceArray(ColorInstanceID, 2, Colors, 3);

		IsInitialized = true;
	}

	if (ShaderForParticle != nullptr && Sphere != nullptr && ParticleCount > 0)
	{
		Real ColorB = 0.0;
		ParticleCount = vParticlePos.size() / 3;
		std::vector<glm::vec3> Translations(ParticleCount);
		for (int i = 0; i < ParticleCount; i++)
		{
			Translations[i] = glm::vec3(vParticlePos[i * 3], vParticlePos[i * 3 + 1], vParticlePos[i * 3 + 2]);
		}
		Sphere->setInstanceArray(OffsetInstanceID, 1, Translations, 3);

		std::vector<glm::vec3> Colors(ParticleCount);
		for (int i = 0; i < ParticleCount; i++)
		{
			Real VelLength = glm::length(glm::vec3(vParticleVel[i * 3], vParticleVel[i * 3 + 1], vParticleVel[i * 3 + 2]));
			Real ClampVel = std::clamp<Real>(VelLength, ColorMinScalar, ColorMaxScalar);

			if (i < vPFParticleSize) ColorB = 1.0;
			else ColorB = 1.0;
			Colors[i] = glm::vec3(
				ClampVel / (ColorMaxScalar - ColorMinScalar),
				ClampVel / (ColorMaxScalar - ColorMinScalar),
				ColorB
			);
		}
		Sphere->setInstanceArray(ColorInstanceID, 2, Colors, 3);

		ShaderForParticle->use();
		ShaderForParticle->setMat4("model", vModel);
		ShaderForParticle->setMat4("view", vView);
		ShaderForParticle->setMat4("projection", vProjection);
		Sphere->drawInstance(*ShaderForParticle, ParticleCount);
	}
}

void drawModel(const std::shared_ptr<graphics::CCamera> vCamera, const glm::mat4& vModel, const glm::mat4& vView, const glm::mat4& vProjection)
{

}

void processInput(GLFWwindow* vWindow)
{
	if (graphics::g_Keys[GLFW_KEY_F] && !graphics::g_KeysPressed[GLFW_KEY_F])
	{
		InitTransform.x += Speed;
	}
	if (graphics::g_Keys[GLFW_KEY_B] && !graphics::g_KeysPressed[GLFW_KEY_B])
	{
		InitTransform.x -= Speed;
	}
	if (graphics::g_Keys[GLFW_KEY_R] && !graphics::g_KeysPressed[GLFW_KEY_R])
	{
		InitRotation = glm::rotate(InitRotation, glm::radians(1.0f), glm::vec3(0, 1, 0));
	}
}