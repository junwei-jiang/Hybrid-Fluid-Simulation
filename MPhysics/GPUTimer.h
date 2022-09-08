#pragma once
#include "Common.h"

struct SGpuTimer
{
	cudaEvent_t m_Start;
	cudaEvent_t m_Stop;

	SGpuTimer()
	{
		cudaEventCreate(&m_Start);
		cudaEventCreate(&m_Stop);
	}

	~SGpuTimer()
	{
		cudaEventDestroy(m_Start);
		cudaEventDestroy(m_Stop);
	}

	void Start()
	{
		cudaEventRecord(m_Start, 0);
	}

	void Stop()
	{
		cudaEventRecord(m_Stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(m_Stop);
		cudaEventElapsedTime(&elapsed, m_Start, m_Stop);
		return elapsed;
	}
};
