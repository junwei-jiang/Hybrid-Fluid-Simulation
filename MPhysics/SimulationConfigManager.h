#pragma once
#include "Common.h"
#include "Manager.h"

class CSimulationConfigManager
{
	Manager(CSimulationConfigManager)

public:
	void setG(Real vInput);
	void setTimeStep(Real vInput);
	void setMaxTimeStep(Real vInput);
	void setMinTimeStep(Real vInput);
	void setSimualtionRange(SAABB vSimualtionRange);

	Real getG() const;
	Real getTimeStep() const;
	SAABB getSimualtionRange() const;

private:
	Real m_G = static_cast<Real>(9.8);
	Real m_TimeStep = static_cast<Real>(0.001);

	SAABB m_SimualtionRange;
};

