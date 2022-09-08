#pragma once
#include "CuDenseVector.h"
#include "CuSparseMatrix.h"
#include "CuMatrixFreePCG.h"
#include "CellCenteredScalarField.h"
#include "CellCenteredVectorField.h"
#include "FaceCenteredVectorField.h"

typedef struct SFdmLinearSystem
{
	Vector3i Resolution;
	Vector3 Spacing;
	thrust::device_vector<Real> FdmMatrixA;
	CCuDenseVector FdmVectorx;
	CCuDenseVector FdmVectorb;
}SFdmLinearSystem;

class CPressureSolver
{
public:
	CPressureSolver();
	CPressureSolver(const Vector3i& vResolution, const Vector3& vOrigin = Vector3(0, 0, 0), const Vector3& vSpacing = Vector3(1, 1, 1));
	~CPressureSolver();

	const thrust::device_vector<Real>& getConstFdmMatrixA() const;
	thrust::device_vector<Real>& getFdmMatrixA();
	const CCuDenseVector& getConstVectorx() const;
	CCuDenseVector& getVectorx();
	const CCuDenseVector& getConstVectorb() const;
	CCuDenseVector& getVectorb();
	const CCellCenteredScalarField& getMarkers() const;

	void solvePressure
	(
		CFaceCenteredVectorField& vioFluidVelField,
		Real vDeltaT,
		const CFaceCenteredVectorField& vSolidVelField,
		const CCellCenteredScalarField& vSolidSDFField,
		const CCellCenteredScalarField& vFluidSDFField 
	);

	void solvePressure
	(
		CFaceCenteredVectorField& vioFluidVelField,
		Real vDeltaT,
		const CFaceCenteredVectorField& vSolidVelField,
		const CCellCenteredScalarField& vSolidSDFField,
		const thrust::device_vector<Real>& vParticlesPos
	);

	void resizePressureSolver(const Vector3i& vResolution, const Vector3& vOrigin = Vector3(0, 0, 0), const Vector3& vSpacing = Vector3(1, 1, 1));

private:
	void __buildMarkers(const CCellCenteredScalarField& vSolidSDFField, const thrust::device_vector<Real>& vParticlesPos);
	void __buildMarkers(const CCellCenteredScalarField& vSolidSDFField, const CCellCenteredScalarField& vFluidSDFField);
	void __buildSystem(Real vDeltaT, const CFaceCenteredVectorField& vInputVelField, const CFaceCenteredVectorField& vSolidVelField);
	void __buildMatrix(Real vDeltaT);
	void __buildVector(const CFaceCenteredVectorField& vInputVelField, const CFaceCenteredVectorField& vSolidVelField);
	void __applyPressureGradient(Real vDeltaT, CFaceCenteredVectorField& vioFluidVelField, const CFaceCenteredVectorField& vSolidVelField);

	Real m_density = 1.0;
	SFdmLinearSystem m_FdmLinearSystem;
	unique_ptr<CCuMatrixFreePCG> m_MartixFreeCGSolver = nullptr;

	CCellCenteredScalarField m_MarkersField;
	CCellCenteredScalarField m_VelDivergenceField;
};

typedef std::shared_ptr<CPressureSolver> PressureSolverPtr;
