#include "SemiLagrangian.h"

CSemiLagrangian::CSemiLagrangian()
{

}

CSemiLagrangian::CSemiLagrangian
(
	const Vector3i& vResolution, 
	const Vector3& vOrigin, 
	const Vector3& vSpacing
)
{
	resizeSemiLagrangian(vResolution, vOrigin, vSpacing);
}

CSemiLagrangian::~CSemiLagrangian()
{

}

void CSemiLagrangian::resizeSemiLagrangian(const Vector3i& vResolution, const Vector3& vOrigin, const Vector3& vSpacing)
{
	Vector3i vResX = Vector3i(vResolution.x + 1, vResolution.y, vResolution.z);
	Vector3i vResY = Vector3i(vResolution.x, vResolution.y + 1, vResolution.z);
	Vector3i vResZ = Vector3i(vResolution.x, vResolution.y, vResolution.z + 1);

	SizeT Size = vResolution.x * vResolution.y * vResolution.z;
	SizeT SizeX = vResX.x * vResX.y * vResX.z;
	SizeT SizeY = vResY.x * vResY.y * vResY.z;
	SizeT SizeZ = vResZ.x * vResZ.y * vResZ.z;

	vector<Real> CCPosFieldDataX(Size);
	vector<Real> CCPosFieldDataY(Size);
	vector<Real> CCPosFieldDataZ(Size);
	vector<Real> FCPosFieldDataXX(SizeX);
	vector<Real> FCPosFieldDataXY(SizeX);
	vector<Real> FCPosFieldDataXZ(SizeX);
	vector<Real> FCPosFieldDataYX(SizeY);
	vector<Real> FCPosFieldDataYY(SizeY);
	vector<Real> FCPosFieldDataYZ(SizeY);
	vector<Real> FCPosFieldDataZX(SizeZ);
	vector<Real> FCPosFieldDataZY(SizeZ);
	vector<Real> FCPosFieldDataZZ(SizeZ);

	for (int i = 0; i < vResolution.z; i++)
	{
		for (int j = 0; j < vResolution.y; j++)
		{
			for (int k = 0; k < vResolution.x; k++)
			{
				CCPosFieldDataX[i * vResolution.x * vResolution.y + j * vResolution.x + k] = (k + static_cast<Real>(0.5)) * vSpacing.x + vOrigin.x;
				CCPosFieldDataY[i * vResolution.x * vResolution.y + j * vResolution.x + k] = (j + static_cast<Real>(0.5)) * vSpacing.y + vOrigin.y;
				CCPosFieldDataZ[i * vResolution.x * vResolution.y + j * vResolution.x + k] = (i + static_cast<Real>(0.5)) * vSpacing.z + vOrigin.z;
			}
		}
	}

	for (int i = 0; i < vResX.z; i++)
	{
		for (int j = 0; j < vResX.y; j++)
		{
			for (int k = 0; k < vResX.x; k++)
			{
				FCPosFieldDataXX[i * vResX.x * vResX.y + j * vResX.x + k] = k * vSpacing.x + vOrigin.x;
				FCPosFieldDataXY[i * vResX.x * vResX.y + j * vResX.x + k] = (j + static_cast<Real>(0.5)) * vSpacing.y + vOrigin.y;
				FCPosFieldDataXZ[i * vResX.x * vResX.y + j * vResX.x + k] = (i + static_cast<Real>(0.5)) * vSpacing.z + vOrigin.z;
			}
		}
	}

	for (int i = 0; i < vResY.z; i++)
	{
		for (int j = 0; j < vResY.y; j++)
		{
			for (int k = 0; k < vResY.x; k++)
			{
				FCPosFieldDataYX[i * vResY.x * vResY.y + j * vResY.x + k] = (k + static_cast<Real>(0.5)) * vSpacing.x + vOrigin.x;
				FCPosFieldDataYY[i * vResY.x * vResY.y + j * vResY.x + k] = j * vSpacing.y + vOrigin.y;
				FCPosFieldDataYZ[i * vResY.x * vResY.y + j * vResY.x + k] = (i + static_cast<Real>(0.5)) * vSpacing.z + vOrigin.z;
			}
		}
	}

	for (int i = 0; i < vResZ.z; i++)
	{
		for (int j = 0; j < vResZ.y; j++)
		{
			for (int k = 0; k < vResZ.x; k++)
			{
				FCPosFieldDataZX[i * vResZ.x * vResZ.y + j * vResZ.x + k] = (k + static_cast<Real>(0.5)) * vSpacing.x + vOrigin.x;
				FCPosFieldDataZY[i * vResZ.x * vResZ.y + j * vResZ.x + k] = (j + static_cast<Real>(0.5)) * vSpacing.y + vOrigin.y;
				FCPosFieldDataZZ[i * vResZ.x * vResZ.y + j * vResZ.x + k] = i * vSpacing.z + vOrigin.z;
			}
		}
	}

	m_AdvectionInputPointPosFieldCC.resize(vResolution, vOrigin, vSpacing, CCPosFieldDataX.data(), CCPosFieldDataY.data(), CCPosFieldDataZ.data());
	m_AdvectionOutputPointPosFieldCC = m_AdvectionInputPointPosFieldCC;
	m_AdvectionInputPointPosXFieldFC.resize(vResX, vOrigin, vSpacing, FCPosFieldDataXX.data(), FCPosFieldDataXY.data(), FCPosFieldDataXZ.data());
	m_AdvectionInputPointPosYFieldFC.resize(vResY, vOrigin, vSpacing, FCPosFieldDataYX.data(), FCPosFieldDataYY.data(), FCPosFieldDataYZ.data());;
	m_AdvectionInputPointPosZFieldFC.resize(vResZ, vOrigin, vSpacing, FCPosFieldDataZX.data(), FCPosFieldDataZY.data(), FCPosFieldDataZZ.data());;
	m_AdvectionOutputPointPosXFieldFC = m_AdvectionInputPointPosXFieldFC;
	m_AdvectionOutputPointPosYFieldFC = m_AdvectionInputPointPosYFieldFC;
	m_AdvectionOutputPointPosZFieldFC = m_AdvectionInputPointPosZFieldFC;

	m_BackTraceInputPointVelField.resize(vResolution);
	m_BackTraceMidPointPosField.resize(vResolution);
	m_BackTraceMidPointVelField.resize(vResolution);
	m_BackTraceTwoThirdsPointPosField.resize(vResolution);
	m_BackTraceTwoThirdsPointVelField.resize(vResolution);

	m_BackTraceInputPointVelFieldX.resize(vResX);
	m_BackTraceMidPointPosFieldX.resize(vResX);
	m_BackTraceMidPointVelFieldX.resize(vResX);
	m_BackTraceTwoThirdsPointPosFieldX.resize(vResX);
	m_BackTraceTwoThirdsPointVelFieldX.resize(vResX);

	m_BackTraceInputPointVelFieldY.resize(vResY);
	m_BackTraceMidPointPosFieldY.resize(vResY);
	m_BackTraceMidPointVelFieldY.resize(vResY);
	m_BackTraceTwoThirdsPointPosFieldY.resize(vResY);
	m_BackTraceTwoThirdsPointVelFieldY.resize(vResY);

	m_BackTraceInputPointVelFieldZ.resize(vResZ);
	m_BackTraceMidPointPosFieldZ.resize(vResZ);
	m_BackTraceMidPointVelFieldZ.resize(vResZ);
	m_BackTraceTwoThirdsPointPosFieldZ.resize(vResZ);
	m_BackTraceTwoThirdsPointVelFieldZ.resize(vResZ);

	resizeAdvectionSolver(vResolution);
}

void CSemiLagrangian::advect
(
	const CCellCenteredScalarField& vInputField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CCellCenteredScalarField& voOutputField,
	EAdvectionAccuracy vEAdvectionAccuracy,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	if (m_IsInit)
	{
		backTrace(m_AdvectionInputPointPosFieldCC, vVelocityField, vDeltaT, m_AdvectionOutputPointPosFieldCC, vEAdvectionAccuracy);
		vInputField.sampleField(m_AdvectionOutputPointPosFieldCC, voOutputField, m_SamplingAlg);
	}
}

void CSemiLagrangian::advect
(
	const CCellCenteredVectorField& vInputField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CCellCenteredVectorField& voOutputField,
	EAdvectionAccuracy vEAdvectionAccuracy,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	if (m_IsInit)
	{
		backTrace(m_AdvectionInputPointPosFieldCC, vVelocityField, vDeltaT, m_AdvectionOutputPointPosFieldCC, vEAdvectionAccuracy);
		vInputField.sampleField(m_AdvectionOutputPointPosFieldCC, voOutputField, m_SamplingAlg);
	}
}

void CSemiLagrangian::advect
(
	const CFaceCenteredVectorField& vInputField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CFaceCenteredVectorField& voOutputField,
	EAdvectionAccuracy vEAdvectionAccuracy,
	const CCellCenteredScalarField& vBoundarysSDF
)
{
	if (m_IsInit)
	{
		backTrace(m_AdvectionInputPointPosXFieldFC, vVelocityField, vDeltaT, m_AdvectionOutputPointPosXFieldFC, vEAdvectionAccuracy);
		backTrace(m_AdvectionInputPointPosYFieldFC, vVelocityField, vDeltaT, m_AdvectionOutputPointPosYFieldFC, vEAdvectionAccuracy);
		backTrace(m_AdvectionInputPointPosZFieldFC, vVelocityField, vDeltaT, m_AdvectionOutputPointPosZFieldFC, vEAdvectionAccuracy);
		vInputField.sampleField(m_AdvectionOutputPointPosXFieldFC, m_AdvectionOutputPointPosYFieldFC, m_AdvectionOutputPointPosZFieldFC, voOutputField, m_SamplingAlg);
	}
}

void CSemiLagrangian::backTrace
(
	const CCellCenteredVectorField& vInputPosField,
	const CFaceCenteredVectorField& vVelocityField,
	Real vDeltaT,
	CCellCenteredVectorField& voOutputPosField,
	EAdvectionAccuracy vEAdvectionAccuracy
)
{
	if (m_IsInit)
	{
		if (vInputPosField.getResolution() == m_Resolution)
		{
			_ASSERTE(voOutputPosField.getResolution() == vInputPosField.getResolution());

			voOutputPosField = vInputPosField;

			vVelocityField.sampleField(vInputPosField, m_BackTraceInputPointVelField, m_SamplingAlg);

			if (vEAdvectionAccuracy == EAdvectionAccuracy::RK1)
			{
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelField, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK2)
			{
				m_BackTraceMidPointPosField = vInputPosField;
				m_BackTraceMidPointPosField.plusAlphaX(m_BackTraceInputPointVelField, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosField, m_BackTraceMidPointVelField, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelField, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK3)
			{
				m_BackTraceMidPointPosField = vInputPosField;
				m_BackTraceMidPointPosField.plusAlphaX(m_BackTraceInputPointVelField, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosField, m_BackTraceMidPointVelField, m_SamplingAlg);
				m_BackTraceTwoThirdsPointPosField = vInputPosField;
				m_BackTraceTwoThirdsPointPosField.plusAlphaX(m_BackTraceMidPointVelField, static_cast<Real>(-0.75) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceTwoThirdsPointPosField, m_BackTraceTwoThirdsPointVelField, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelField, -static_cast<Real>(2.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelField, -static_cast<Real>(3.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceTwoThirdsPointVelField, -static_cast<Real>(4.0 / 9.0) * vDeltaT);
			}
			else
			{

			}
		}
		else if (vInputPosField.getResolution() == m_Resolution + Vector3i(1, 0, 0))
		{
			_ASSERTE(voOutputPosField.getResolution() == vInputPosField.getResolution());

			voOutputPosField = vInputPosField;

			vVelocityField.sampleField(vInputPosField, m_BackTraceInputPointVelFieldX, m_SamplingAlg);

			if (vEAdvectionAccuracy == EAdvectionAccuracy::RK1)
			{
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelFieldX, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK2)
			{
				m_BackTraceMidPointPosFieldX = vInputPosField;
				m_BackTraceMidPointPosFieldX.plusAlphaX(m_BackTraceInputPointVelFieldX, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosFieldX, m_BackTraceMidPointVelFieldX, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelFieldX, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK3)
			{
				m_BackTraceMidPointPosFieldX = vInputPosField;
				m_BackTraceMidPointPosFieldX.plusAlphaX(m_BackTraceInputPointVelFieldX, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosFieldX, m_BackTraceMidPointVelFieldX, m_SamplingAlg);
				m_BackTraceTwoThirdsPointPosFieldX = vInputPosField;
				m_BackTraceTwoThirdsPointPosFieldX.plusAlphaX(m_BackTraceMidPointVelFieldX, static_cast<Real>(-0.75) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceTwoThirdsPointPosFieldX, m_BackTraceTwoThirdsPointVelFieldX, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelFieldX, -static_cast<Real>(2.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelFieldX, -static_cast<Real>(3.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceTwoThirdsPointVelFieldX, -static_cast<Real>(4.0 / 9.0) * vDeltaT);
			}
			else
			{

			}
		}
		else if (vInputPosField.getResolution() == m_Resolution + Vector3i(0, 1, 0))
		{
			_ASSERTE(voOutputPosField.getResolution() == vInputPosField.getResolution());

			voOutputPosField = vInputPosField;

			vVelocityField.sampleField(vInputPosField, m_BackTraceInputPointVelFieldY, m_SamplingAlg);

			if (vEAdvectionAccuracy == EAdvectionAccuracy::RK1)
			{
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelFieldY, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK2)
			{
				m_BackTraceMidPointPosFieldY = vInputPosField;
				m_BackTraceMidPointPosFieldY.plusAlphaX(m_BackTraceInputPointVelFieldY, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosFieldY, m_BackTraceMidPointVelFieldY, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelFieldY, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK3)
			{
				m_BackTraceMidPointPosFieldY = vInputPosField;
				m_BackTraceMidPointPosFieldY.plusAlphaX(m_BackTraceInputPointVelFieldY, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosFieldY, m_BackTraceMidPointVelFieldY, m_SamplingAlg);
				m_BackTraceTwoThirdsPointPosFieldY = vInputPosField;
				m_BackTraceTwoThirdsPointPosFieldY.plusAlphaX(m_BackTraceMidPointVelFieldY, static_cast<Real>(-0.75) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceTwoThirdsPointPosFieldY, m_BackTraceTwoThirdsPointVelFieldY, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelFieldY, -static_cast<Real>(2.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelFieldY, -static_cast<Real>(3.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceTwoThirdsPointVelFieldY, -static_cast<Real>(4.0 / 9.0) * vDeltaT);
			}
			else
			{

			}
		}
		else if (vInputPosField.getResolution() == m_Resolution + Vector3i(0, 0, 1))
		{
			_ASSERTE(voOutputPosField.getResolution() == vInputPosField.getResolution());

			voOutputPosField = vInputPosField;

			vVelocityField.sampleField(vInputPosField, m_BackTraceInputPointVelFieldZ, m_SamplingAlg);

			if (vEAdvectionAccuracy == EAdvectionAccuracy::RK1)
			{
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelFieldZ, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK2)
			{
				m_BackTraceMidPointPosFieldZ = vInputPosField;
				m_BackTraceMidPointPosFieldZ.plusAlphaX(m_BackTraceInputPointVelFieldZ, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosFieldZ, m_BackTraceMidPointVelFieldZ, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelFieldZ, -vDeltaT);
			}
			else if (vEAdvectionAccuracy == EAdvectionAccuracy::RK3)
			{
				m_BackTraceMidPointPosFieldZ = vInputPosField;
				m_BackTraceMidPointPosFieldZ.plusAlphaX(m_BackTraceInputPointVelFieldZ, static_cast<Real>(-0.5) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceMidPointPosFieldZ, m_BackTraceMidPointVelFieldZ, m_SamplingAlg);
				m_BackTraceTwoThirdsPointPosFieldZ = vInputPosField;
				m_BackTraceTwoThirdsPointPosFieldZ.plusAlphaX(m_BackTraceMidPointVelFieldZ, static_cast<Real>(-0.75) * vDeltaT);
				vVelocityField.sampleField(m_BackTraceTwoThirdsPointPosFieldZ, m_BackTraceTwoThirdsPointVelFieldZ, m_SamplingAlg);
				voOutputPosField.plusAlphaX(m_BackTraceInputPointVelFieldZ, -static_cast<Real>(2.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceMidPointVelFieldZ, -static_cast<Real>(3.0 / 9.0) * vDeltaT);
				voOutputPosField.plusAlphaX(m_BackTraceTwoThirdsPointVelFieldZ, -static_cast<Real>(4.0 / 9.0) * vDeltaT);
			}
			else
			{

			}
		}
		else
		{
			return;
		}
	}
}