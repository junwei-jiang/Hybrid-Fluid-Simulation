#include "CuSparsePCG.h"
#include "ThrustWapper.cuh"
#include "CudaContextManager.h"

CCuSparsePCG::~CCuSparsePCG()
{
}

void CCuSparsePCG::init(UInt vColSize, UInt vRowSize)
{
	m_ColSize = vColSize;
	m_RowSize = vRowSize;
	m_ZM1.resize(vColSize, 0);
	m_ZM2.resize(vColSize, 0);
	m_R.resize(vColSize, 0);
	m_RM1.resize(vColSize, 0);
	m_RM2.resize(vColSize, 0);
	m_P.resize(vColSize, 0);
	m_TempVector.resize(vColSize, 0);
	m_AP.resize(vColSize, 0);
}

void CCuSparsePCG::freshPCacheSize(UInt vNonZeroSize)
{
	m_NonZeroSize = vNonZeroSize;
	resizeDeviceVector(m_ILU0Value, vNonZeroSize, 0);
	resizeDeviceVector(m_NonZeroCache, vNonZeroSize, 0);
	resizeDeviceVector(m_RowIndicesCache, vNonZeroSize, 0);
	resizeDeviceVector(m_ColIndicesCache, vNonZeroSize, 0);

	Real* NNZ = getRawDevicePointerReal(m_NonZeroCache);
	Int* Col = getRawDevicePointerInt(m_ColIndicesCache);
	Int* Row = getRawDevicePointerInt(m_RowIndicesCache);

	CHECK_CUSPARSE(cusparseCreateMatDescr(&DescrA));
	CHECK_CUSPARSE(cusparseSetMatType(DescrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(DescrA, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseCreateCsr(
		&MatA, m_RowSize, m_ColSize, vNonZeroSize, Row, Col, NNZ, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_REAL_TYPE));

	/* Create ILU(0) info object */
	CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&InfoILU));

	/* Create L factor descriptor and triangular solve info */
	CHECK_CUSPARSE(cusparseCreateMatDescr(&DescrL));
	CHECK_CUSPARSE(cusparseSetMatType(DescrL, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(DescrL, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatFillMode(DescrL, CUSPARSE_FILL_MODE_LOWER));
	CHECK_CUSPARSE(cusparseSetMatDiagType(DescrL, CUSPARSE_DIAG_TYPE_UNIT));
	CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&InfoL));

	/* Create U factor descriptor and triangular solve info */
	CHECK_CUSPARSE(cusparseCreateMatDescr(&DescrU));
	CHECK_CUSPARSE(cusparseSetMatType(DescrU, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(DescrU, CUSPARSE_INDEX_BASE_ZERO));
	CHECK_CUSPARSE(cusparseSetMatFillMode(DescrU, CUSPARSE_FILL_MODE_UPPER));
	CHECK_CUSPARSE(cusparseSetMatDiagType(
		DescrU, CUSPARSE_DIAG_TYPE_NON_UNIT));
	CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&InfoU));


	/* Allocate workspace for cuSPARSE */
	size_t bufferSize = 0;
	size_t tmp = 0;
	int stmp = 0;
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(
		CCudaContextManager::getInstance().getCusparseHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &One, MatA,
		m_P.getCuDenseVectorDescr(), &Zero, m_AP.getCuDenseVectorDescr(), CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
		&tmp));
	if (tmp > bufferSize) {
		bufferSize = stmp;
	}
	CHECK_CUSPARSE(cusparseScsrilu02_bufferSize(
		CCudaContextManager::getInstance().getCusparseHandle(), m_ColSize, vNonZeroSize, DescrA, NNZ, Row, Col, InfoILU, &stmp));
	if (stmp > bufferSize) {
		bufferSize = stmp;
	}
	CHECK_CUSPARSE(cusparseScsrsv2_bufferSize(
		CCudaContextManager::getInstance().getCusparseHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, m_ColSize, vNonZeroSize, DescrL, NNZ,
		Row, Col, InfoL, &stmp));
	if (stmp > bufferSize) {
		bufferSize = stmp;
	}
	CHECK_CUSPARSE(cusparseScsrsv2_bufferSize(
		CCudaContextManager::getInstance().getCusparseHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, m_ColSize, vNonZeroSize, DescrU, NNZ,
		Row, Col, InfoU, &stmp));
	if (stmp > bufferSize) {
		bufferSize = stmp;
	}
	CHECK_CUDA(cudaMalloc(&m_PCGILU0Buffer, bufferSize));
}

bool CCuSparsePCG::solvePCGILU0(const CCuDenseVector & vVectorb, CCuDenseVector & voX)
{
	Real* NNZ = getRawDevicePointerReal(m_NonZeroCache);
	Int* Col = getRawDevicePointerInt(m_ColIndicesCache);
	Int* Row = getRawDevicePointerInt(m_RowIndicesCache);

	UInt K = 0;
	m_R = vVectorb;
	Real R1 = m_R * m_R;

	/* Perform analysis for ILU(0) */
	CHECK_CUSPARSE(cusparseScsrilu02_analysis(
		CCudaContextManager::getInstance().getCusparseHandle(),
		m_ColSize, m_NonZeroSize, DescrA, NNZ, Row, Col, InfoILU,
		CUSPARSE_SOLVE_POLICY_USE_LEVEL, m_PCGILU0Buffer
	));

	/* Copy A data to ILU(0) vals as input*/
	CHECK_CUDA(cudaMemcpy(
		getRawDevicePointerReal(m_ILU0Value), NNZ, m_NonZeroSize * sizeof(Real), cudaMemcpyDeviceToDevice
	));

	/* generate the ILU(0) factors */
	CHECK_CUSPARSE(cusparseScsrilu02(
		CCudaContextManager::getInstance().getCusparseHandle(),
		m_ColSize, m_NonZeroSize, DescrA, getRawDevicePointerReal(m_ILU0Value), Row, Col, InfoILU,
		CUSPARSE_SOLVE_POLICY_USE_LEVEL, m_PCGILU0Buffer
	));

	/* perform triangular solve analysis */
	CHECK_CUSPARSE(cusparseScsrsv2_analysis(
		CCudaContextManager::getInstance().getCusparseHandle(),
		CUSPARSE_OPERATION_NON_TRANSPOSE, m_ColSize, m_NonZeroSize, DescrL,
		getRawDevicePointerReal(m_ILU0Value), Row, Col, InfoL, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
		m_PCGILU0Buffer));

	CHECK_CUSPARSE(cusparseScsrsv2_analysis(
		CCudaContextManager::getInstance().getCusparseHandle(),
		CUSPARSE_OPERATION_NON_TRANSPOSE, m_ColSize, m_NonZeroSize, DescrU,
		getRawDevicePointerReal(m_ILU0Value), Row, Col, InfoU, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
		m_PCGILU0Buffer));

	voX.setZero();

	while (R1 > m_Threshold * m_Threshold && K <= m_IterationNum)
	{
		CHECK_CUSPARSE(cusparseScsrsv2_solve(CCudaContextManager::getInstance().getCusparseHandle(),
			CUSPARSE_OPERATION_NON_TRANSPOSE, m_ColSize, m_NonZeroSize, &One, DescrL,
			getRawDevicePointerReal(m_ILU0Value), Row, Col, InfoL, m_R.getVectorValueGPUPtr(), m_TempVector.getVectorValueGPUPtr(),
			CUSPARSE_SOLVE_POLICY_USE_LEVEL, m_PCGILU0Buffer));
		CHECK_CUSPARSE(cusparseScsrsv2_solve(CCudaContextManager::getInstance().getCusparseHandle(),
			CUSPARSE_OPERATION_NON_TRANSPOSE, m_ColSize, m_NonZeroSize, &One, DescrU,
			getRawDevicePointerReal(m_ILU0Value), Row, Col, InfoU, m_TempVector.getVectorValueGPUPtr(), m_ZM1.getVectorValueGPUPtr(),
			CUSPARSE_SOLVE_POLICY_USE_LEVEL, m_PCGILU0Buffer));
		K++;
		if (K == 1)
		{
			m_P = m_ZM1;
		}
		else
		{
			Real Numerator = m_R * m_ZM1;
			Real Denominator = m_RM2 * m_ZM1;
			Real Beta = Numerator / Denominator;
			m_P *= Beta;
			m_P += m_ZM1;
		}

		CHECK_CUSPARSE(cusparseSpMV(
			CCudaContextManager::getInstance().getCusparseHandle(),
			CUSPARSE_OPERATION_NON_TRANSPOSE, &One, MatA,
			m_P.getCuDenseVectorDescr(), &Zero, m_AP.getCuDenseVectorDescr(), CUDA_REAL_TYPE, CUSPARSE_SPMV_ALG_DEFAULT,
			m_PCGILU0Buffer
		));
		Real Numerator = m_R * m_ZM1;
		Real Denominator = m_P * m_AP;
		Real Alpha = Numerator / Denominator;
		voX.plusAlphaX(m_P, Alpha);
		m_RM2 = m_R;
		m_ZM2 = m_ZM1;
		Real NegAlpha = -Alpha;
		m_R.plusAlphaX(m_AP, NegAlpha);
		R1 = m_R * m_R;
	}

	return K == 0;
}

void CCuSparsePCG::setIterationNum(UInt vIterationNum)
{
	m_IterationNum = vIterationNum;
}

void CCuSparsePCG::setThreshold(Real vThreshold)
{
	m_Threshold = vThreshold;
}

Real CCuSparsePCG::getThreshold()
{
	return m_Threshold;
}

Real * CCuSparsePCG::getNonZeroCacheGPUPtr()
{
	return getRawDevicePointerReal(m_NonZeroCache);
}

Int * CCuSparsePCG::getRowIndicesCacheGPUPtr()
{
	return getRawDevicePointerInt(m_RowIndicesCache);
}

Int * CCuSparsePCG::getColIndicesCacheGPUPtr()
{
	return getRawDevicePointerInt(m_ColIndicesCache);
}
