#pragma once
#include "Common.h"
#include "CellCenteredScalarField.h"
#include "FaceCenteredVectorField.h"

class CEulerParticles
{
public:
	CEulerParticles();
	CEulerParticles(UInt vNumOfParticles, const Real* vParticlesPos = nullptr, const Real* vParticlesVel = nullptr, const Real* vParticlesColor = nullptr);
	virtual ~CEulerParticles();

	void resize(UInt vNumOfParticles, const Real* vParticlesPos = nullptr, const Real* vParticlesVel = nullptr, const Real* vParticlesColor = nullptr);

	UInt getNumOfParticles() const;
	const thrust::device_vector<Real>& getParticlesPos() const;
	const thrust::device_vector<Real>& getParticlesVel() const;
	thrust::device_vector<Real>& getParticlesVel();
	const thrust::device_vector<Real>& getParticlesColor() const;

	const Real* getConstParticlesPosGPUPtr() const;
	const Real* getConstParticlesVelGPUPtr() const;
	const Real* getConstParticlesColorGPUPtr() const;
	Real* getParticlesPosGPUPtr();
	Real* getParticlesVelGPUPtr();
	Real* getParticlesColorGPUPtr();

	void setParticlesPos(const Real* vParticlesPosCPUPtr);
	void setParticlesVel(const Real* vParticlesVelXCPUPtr);
	void setParticlesColor(const Real* vParticlesColorCPUPtr);

	void generateParticlesInFluid(const CCellCenteredScalarField& vFluidSDF, const CCellCenteredScalarField& vSolidSDF, UInt vNumOfPerGrid);
	void statisticalFluidDensity(CCellCenteredScalarField& voFluidDensityField);

	void transferParticlesScalarValue2Field(CCellCenteredScalarField& voScalarField, CCellCenteredScalarField& voWeightField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	void transferParticlesVectorValue2Field(CCellCenteredVectorField& voVectorField, CCellCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	void transferParticlesVectorValue2Field(CFaceCenteredVectorField& voVectorField, CFaceCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	void transferScalarField2Particles(const CCellCenteredScalarField& vScalarField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	void transferVectorField2Particles(const CCellCenteredVectorField& vVectorField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	void transferVectorField2Particles(const CFaceCenteredVectorField& vVectorField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	
	void transferParticlesColor2Field(CCellCenteredVectorField& voColorField, CCellCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	void transferParticlesVel2Field(CFaceCenteredVectorField& voVelField, CFaceCenteredVectorField& voWeightField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);
	void transferVelField2Particles(const CFaceCenteredVectorField& vVelField, EPGTransferAlgorithm vTransferAlg = EPGTransferAlgorithm::LINEAR);

	void advectParticlesInVelField(const CFaceCenteredVectorField& vVelField, Real vDeltaT, Real vCFLNumber, const CCellCenteredScalarField& vBoundarysSDF = CCellCenteredScalarField(), EAdvectionAccuracy vAdvectionAccuracy = EAdvectionAccuracy::RK3, ESamplingAlgorithm vSamplingAlg = ESamplingAlgorithm::TRILINEAR);

private:
	UInt m_NumOfParticles;

	thrust::device_vector<Real> m_ParticlesPos;
	thrust::device_vector<Real> m_ParticlesVel;
	thrust::device_vector<Real> m_ParticlesColor;
	thrust::device_vector<Real> m_ParticlesScalarValue;
	thrust::device_vector<Real> m_ParticlesVectorValue;

	thrust::device_vector<Real> m_ParticlesMidPos;
	thrust::device_vector<Real> m_ParticlesThreeFourthsPos;
	thrust::device_vector<Real> m_VelFieldCurPosVel;
	thrust::device_vector<Real> m_VelFieldMidPosVel;
	thrust::device_vector<Real> m_VelFieldThreeFourthsPosVel;

	void __fixParticlesPosWithBoundarys(const CCellCenteredScalarField& vBoundarysSDF);
};
