#include "SceneManager.h"
#include "SimulationConfigManager.h"

#include <sstream>

void CSceneManager::loadScene(const std::string& vSceneFile)
{
	boost::property_tree::ptree SceneData;
	if (!std::filesystem::exists(vSceneFile))
	{
		stringstream ss(vSceneFile);
		boost::property_tree::json_parser::read_json(ss, SceneData);
	}
	else
	{
		boost::property_tree::json_parser::read_json(vSceneFile, SceneData);
	}

	//Simulation Config数据
	boost::property_tree::ptree SimulationConfigData = SceneData.get_child("SimulationConfig");
	CSimulationConfigManager::getInstance().setTimeStep(SimulationConfigData.get_child("TimeStep").get_value<Real>());
	CSimulationConfigManager::getInstance().setG(SimulationConfigData.get_child("Gravity").get_value<Real>());
	SAABB SimulationRange;
	SimulationRange.Min.x = SimulationConfigData.get_child("SimulationRange.Start.x").get_value<Real>();
	SimulationRange.Min.y = SimulationConfigData.get_child("SimulationRange.Start.y").get_value<Real>();
	SimulationRange.Min.z = SimulationConfigData.get_child("SimulationRange.Start.z").get_value<Real>();
	SimulationRange.Max.x = SimulationConfigData.get_child("SimulationRange.End.x").get_value<Real>();
	SimulationRange.Max.y = SimulationConfigData.get_child("SimulationRange.End.y").get_value<Real>();
	SimulationRange.Max.z = SimulationConfigData.get_child("SimulationRange.End.z").get_value<Real>();
	CSimulationConfigManager::getInstance().setSimualtionRange(SimulationRange);

	//加载HybridSimulator
	boost::property_tree::ptree Simulators = SceneData.get_child("HybridSimulator");
	for (auto SimulatorData : Simulators)
	{
		string Name = SimulatorData.second.get_child("Name").get_value<string>();

		if (m_HybridSimulators.count(Name) != 0)
		{
			//throw std::invalid_argument(string(Name) + "已存在！");
			//continue;
			return;
		}

		m_HybridSimulators[Name] = make_shared<CHybridSimulator>(SimulatorData.second);
	}
}

void CSceneManager::freeScene()
{
	for (auto& Simulator : m_HybridSimulators)
	{
		Simulator.second.reset();
		Simulator.second = nullptr;
	}
	m_HybridSimulators.clear();
}

const std::shared_ptr<CHybridSimulator>& CSceneManager::getHybridSimulators(const string& vName)
{
	if (m_HybridSimulators.count(vName) == 0)
	{
		throw std::invalid_argument(vName + "不存在！");
		return nullptr;
	}

	return m_HybridSimulators[vName];
}

const map<string, std::shared_ptr<CHybridSimulator>>& CSceneManager::getAllHybridSimulators()
{
	return m_HybridSimulators;
}
