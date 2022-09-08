#include "SimulationConfigManager.h"

void CSimulationConfigManager::setG(Real vInput)
{
	m_G = vInput;
}

void CSimulationConfigManager::setTimeStep(Real vInput)
{
	m_TimeStep = vInput;
}

void CSimulationConfigManager::setSimualtionRange(SAABB vSimualtionRange)
{
	m_SimualtionRange = vSimualtionRange;
}

Real CSimulationConfigManager::getTimeStep() const
{
	return m_TimeStep;
}

SAABB CSimulationConfigManager::getSimualtionRange() const
{
	return m_SimualtionRange;
}

Real CSimulationConfigManager::getG() const
{
	return m_G;
}
