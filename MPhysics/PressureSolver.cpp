#include "PressureSolver.h"
#include "FieldMathTool.cuh"
#include "EulerSolverTool.cuh"
#include "EulerParticlesTool.cuh"

void PressureAxCGProd(const CCuDenseVector& vX, CCuDenseVector& voResult, void* vFdmLinearSystem)
{
	SFdmLinearSystem* FdmLinearSystem = (SFdmLinearSystem*)vFdmLinearSystem;
	fdmMatrixVectorMulInvoker(FdmLinearSystem->Resolution, FdmLinearSystem->FdmMatrixA, vX, voResult);
}

void PressureMinvxCGProd(const CCuDenseVector & vX, CCuDenseVector & voResult, void * vUserMatrix)
{
	voResult = vX;
}

CPressureSolver::CPressureSolver()
{

}

CPressureSolver::CPressureSolver(const Vector3i& vResolution, const Vector3& vOrigin, const Vector3& vSpacing)
{
	resizePressureSolver(vResolution, vOrigin, vSpacing);
}

CPressureSolver::~CPressureSolver()
{

}

const thrust::device_vector<Real>& CPressureSolver::getConstFdmMatrixA() const
{
	return m_FdmLinearSystem.FdmMatrixA;
}

thrust::device_vector<Real>& CPressureSolver::getFdmMatrixA()
{
	return m_FdmLinearSystem.FdmMatrixA;
}

const CCuDenseVector& CPressureSolver::getConstVectorx() const
{
	return m_FdmLinearSystem.FdmVectorx;
}

CCuDenseVector& CPressureSolver::getVectorx()
{
	return m_FdmLinearSystem.FdmVectorx;
}

const CCuDenseVector& CPressureSolver::getConstVectorb() const
{
	return m_FdmLinearSystem.FdmVectorb;
}

CCuDenseVector& CPressureSolver::getVectorb()
{
	return m_FdmLinearSystem.FdmVectorb;
}

const CCellCenteredScalarField& CPressureSolver::getMarkers() const
{
	return m_MarkersField;
}

void CPressureSolver::resizePressureSolver(const Vector3i& vResolution, const Vector3& vOrigin, const Vector3& vSpacing)
{
	m_FdmLinearSystem.Resolution = vResolution;
	m_FdmLinearSystem.Spacing = vSpacing;

	resizeDeviceVector(m_FdmLinearSystem.FdmMatrixA, 4 * vResolution.x * vResolution.y * vResolution.z);
	m_FdmLinearSystem.FdmVectorx.resize(vResolution.x * vResolution.y * vResolution.z);
	m_FdmLinearSystem.FdmVectorb.resize(vResolution.x * vResolution.y * vResolution.z);
	m_FdmLinearSystem.FdmVectorx.setZero();
	m_FdmLinearSystem.FdmVectorb.setZero();

	m_MarkersField.resize(vResolution);
	m_VelDivergenceField.resize(vResolution);
	m_MartixFreeCGSolver = make_unique<CCuMatrixFreePCG>();
	m_MartixFreeCGSolver->init(vResolution.x * vResolution.y * vResolution.z, vResolution.x * vResolution.y * vResolution.z, vResolution.x * vResolution.y * vResolution.z, PressureAxCGProd, PressureMinvxCGProd);
	m_MartixFreeCGSolver->setIterationNum(vResolution.x * vResolution.y * vResolution.z);
}

void CPressureSolver::solvePressure
(
	CFaceCenteredVectorField& vioFluidVelField,
	Real vDeltaT,
	const CFaceCenteredVectorField& vSolidVelField,
	const CCellCenteredScalarField& vSolidSDFField,
	const CCellCenteredScalarField& vFluidSDFField
)
{
	_ASSERT(vioFluidVelField.getResolution() == m_FdmLinearSystem.Resolution);
	_ASSERT(vSolidSDFField.getResolution() == m_FdmLinearSystem.Resolution);
	_ASSERT(vSolidVelField.getResolution() == m_FdmLinearSystem.Resolution);
	_ASSERT(vFluidSDFField.getResolution() == m_FdmLinearSystem.Resolution);
	_ASSERT(vioFluidVelField.getSpacing() == m_FdmLinearSystem.Spacing);
	_ASSERT(vSolidSDFField.getSpacing() == m_FdmLinearSystem.Spacing);
	_ASSERT(vSolidVelField.getSpacing() == m_FdmLinearSystem.Spacing);
	_ASSERT(vFluidSDFField.getSpacing() == m_FdmLinearSystem.Spacing);

	__buildMarkers(vSolidSDFField, vFluidSDFField);
	__buildSystem(vDeltaT, vioFluidVelField, vSolidVelField);
	m_MartixFreeCGSolver->solvePCGInvDia(m_FdmLinearSystem.FdmVectorb, m_FdmLinearSystem.FdmVectorx, &m_FdmLinearSystem);
	__applyPressureGradient(vDeltaT, vioFluidVelField, vSolidVelField);
}

void CPressureSolver::solvePressure
(
	CFaceCenteredVectorField& vioFluidVelField,
	Real vDeltaT,
	const CFaceCenteredVectorField& vSolidVelField,
	const CCellCenteredScalarField& vSolidSDFField,
	const thrust::device_vector<Real>& vParticlesPos
)
{
	_ASSERT(vioFluidVelField.getResolution() == m_FdmLinearSystem.Resolution);
	_ASSERT(vSolidSDFField.getResolution() == m_FdmLinearSystem.Resolution);
	_ASSERT(vSolidVelField.getResolution() == m_FdmLinearSystem.Resolution);
	_ASSERT(vioFluidVelField.getSpacing() == m_FdmLinearSystem.Spacing);
	_ASSERT(vSolidSDFField.getSpacing() == m_FdmLinearSystem.Spacing);
	_ASSERT(vSolidVelField.getSpacing() == m_FdmLinearSystem.Spacing);

	__buildMarkers(vSolidSDFField, vParticlesPos);
	__buildSystem(vDeltaT, vioFluidVelField, vSolidVelField);
	m_MartixFreeCGSolver->solvePCGInvDia(m_FdmLinearSystem.FdmVectorb, m_FdmLinearSystem.FdmVectorx, &m_FdmLinearSystem);
	__applyPressureGradient(vDeltaT, vioFluidVelField, vSolidVelField);
}

void CPressureSolver::__buildMarkers(const CCellCenteredScalarField& vSolidSDFField, const thrust::device_vector<Real>& vParticlesPos)
{
	buildFluidMarkersInvoker(vSolidSDFField, vParticlesPos, m_MarkersField);
}

void CPressureSolver::__buildMarkers(const CCellCenteredScalarField& vSolidSDFField, const CCellCenteredScalarField& vFluidSDFField)
{
	buildFluidMarkersInvoker(vSolidSDFField, vFluidSDFField, m_MarkersField);
}

void CPressureSolver::__buildSystem(Real vDeltaT, const CFaceCenteredVectorField& vInputVelField, const CFaceCenteredVectorField& vSolidVelField)
{
	__buildMatrix(vDeltaT);
	__buildVector(vInputVelField, vSolidVelField);
	m_FdmLinearSystem.FdmVectorx.setZero();
}

void CPressureSolver::__buildMatrix(Real vDeltaT)
{
	Vector3 Scale = Vector3(vDeltaT / (m_density * m_FdmLinearSystem.Spacing.x * m_FdmLinearSystem.Spacing.x), vDeltaT / (m_density * m_FdmLinearSystem.Spacing.y * m_FdmLinearSystem.Spacing.y), vDeltaT / (m_density * m_FdmLinearSystem.Spacing.z * m_FdmLinearSystem.Spacing.z));
	buildPressureFdmMatrixAInvoker(m_FdmLinearSystem.Resolution, Scale, m_MarkersField, m_FdmLinearSystem.FdmMatrixA);
}

void CPressureSolver::__buildVector(const CFaceCenteredVectorField& vInputVelField, const CFaceCenteredVectorField& vSolidVelField)
{
	vInputVelField.divergence(m_VelDivergenceField);
	buildPressureVectorbInvoker(vInputVelField, m_VelDivergenceField, m_MarkersField, vSolidVelField, m_FdmLinearSystem.FdmVectorb);
}

void CPressureSolver::__applyPressureGradient(Real vDeltaT, CFaceCenteredVectorField& vioFluidVelField, const CFaceCenteredVectorField& vSolidVelField)
{
	Vector3 Scale = Vector3(vDeltaT / (m_density * m_FdmLinearSystem.Spacing.x), vDeltaT / (m_density * m_FdmLinearSystem.Spacing.y), vDeltaT / (m_density * m_FdmLinearSystem.Spacing.z));
	applyPressureGradientInvoker(m_FdmLinearSystem.Resolution, Scale, m_MarkersField, m_FdmLinearSystem.FdmVectorx.getVectorValueGPUPtr(), vioFluidVelField, vSolidVelField);
}