#pragma once
#include "Common.h"
#include "Manager.h"
#include "HybridSimulator.h"

class CSceneManager
{
	Manager(CSceneManager)

public:
	void loadScene(const std::string& vSceneFile);
	void freeScene();

	const shared_ptr<CHybridSimulator>& getHybridSimulators(const string& vName);
	const map<string, shared_ptr<CHybridSimulator>>& getAllHybridSimulators();

private:
	map<string, shared_ptr<CHybridSimulator>> m_HybridSimulators;
};