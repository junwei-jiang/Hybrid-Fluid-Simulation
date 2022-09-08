#include "pch.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"
#include "Particle.h"

#include <random>

TEST(ParticleGroup, AddAndReduceParticle)
{
	CParticleGroup Target;
	SAABB Range;
	Range.Min = Vector3(0, 0, 0);
	Range.Max = Vector3(1, 1, 1);
	Target.setParticleRadius(0.025);
	Target.appendParticleBlock(Range);
	UInt Size_backup = Target.getSize();
	UInt Step = 4;
	UInt Num = 5000;
	UInt EmptySize = 100000;

	SGpuTimer Timer;
	Timer.Start();
	Target.appendEmptyParticle(EmptySize);
	Timer.Stop();
	HighLightLog("Append Time", to_string(Timer.Elapsed()));

	thrust::device_ptr<Real> ParticlePosTGPUPtr(Target.getParticlePosGPUPtr());
	thrust::device_ptr<Real> ParticlePrevPosTGPUPtr(Target.getParticleVelGPUPtr());
	thrust::device_ptr<Real> ParticleVelTGPUPtr(Target.getPrevParticlePosGPUPtr());
	thrust::device_ptr<Real> ParticleLiveTimeTGPUPtr(Target.getParticleLiveTimeGPUPtr());

	srand(time(NULL));
	for (UInt i = 0; i < Num; i++)
	{
		ParticlePosPortion(ParticlePosTGPUPtr, Target.getSize() + i * Step, 0) = rand() % 100;
		ParticlePosPortion(ParticlePosTGPUPtr, Target.getSize() + i * Step, 1) = rand() % 100;
		ParticlePosPortion(ParticlePosTGPUPtr, Target.getSize() + i * Step, 2) = rand() % 100;

		ParticlePosPortion(ParticlePrevPosTGPUPtr, Target.getSize() + i * Step, 0) = rand() % 100;
		ParticlePosPortion(ParticlePrevPosTGPUPtr, Target.getSize() + i * Step, 1) = rand() % 100;
		ParticlePosPortion(ParticlePrevPosTGPUPtr, Target.getSize() + i * Step, 2) = rand() % 100;

		ParticlePosPortion(ParticleVelTGPUPtr, Target.getSize() + i * Step, 0) = rand() % 100;
		ParticlePosPortion(ParticleVelTGPUPtr, Target.getSize() + i * Step, 1) = rand() % 100;
		ParticlePosPortion(ParticleVelTGPUPtr, Target.getSize() + i * Step, 2) = rand() % 100;

		ParticlePosPortion(ParticleLiveTimeTGPUPtr, Target.getSize() + i * Step, 0) = rand() % 100;
		ParticlePosPortion(ParticleLiveTimeTGPUPtr, Target.getSize() + i * Step, 1) = rand() % 100;
		ParticlePosPortion(ParticleLiveTimeTGPUPtr, Target.getSize() + i * Step, 2) = rand() % 100;
	}

	Timer.Start();
	Target.reduceEmptyParticle();
	Timer.Stop();
	HighLightLog("Reduce Time", to_string(Timer.Elapsed()));

	ASSERT_EQ(Target.getSize(), Size_backup + Num);
	ASSERT_EQ(Target.getRealSize(), Size_backup + EmptySize);
}

struct SOdd
{
	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
		thrust::get<1>(t) = thrust::get<0>(t) / 3 % 2 == 0;
	}
};

TEST(ParticleGroup, removeParticle)
{
	CParticleGroup Target;
	SAABB Range;
	Range.Min = Vector3(0, 0, 0);
	Range.Max = Vector3(5, 5, 5);
	Target.setParticleRadius(0.025);
	Target.appendParticleBlock(Range);
	UInt Size_backup = Target.getSize();

	thrust::device_vector<bool> FilterMap(Target.getSize() * 3, false);
	thrust::counting_iterator<UInt> First(0);
	thrust::counting_iterator<UInt> Last = First + Target.getSize() * 3;

	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(First, FilterMap.begin())
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(Last, FilterMap.end())
		), 
		SOdd()
	);

	SGpuTimer Timer;
	Timer.Start();
	Target.removeParticles(FilterMap);
	Timer.Stop();
	HighLightLog("Remove Time", to_string(Timer.Elapsed()));

	ASSERT_EQ(Target.getSize(), (UInt)(Size_backup * 0.5));
	ASSERT_EQ(Target.getRealSize(), Size_backup);
}