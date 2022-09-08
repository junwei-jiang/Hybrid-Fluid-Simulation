#pragma once
#include "Common.h"
#include "CuDenseVector.h"
class CCuSparsePCG
{
public:
	CCuSparsePCG() = default;
	~CCuSparsePCG();
	void init(UInt vColSize, UInt vRowSize);	
	void freshPCacheSize(UInt vNonZeroSize);
	bool solvePCGILU0
	(
		const CCuDenseVector & vVectorb,
		CCuDenseVector & voX
	);

	void setIterationNum(UInt vIterationNum);
	void setThreshold(Real vThreshold);
	Real getThreshold();

	Real* getNonZeroCacheGPUPtr();
	Int* getRowIndicesCacheGPUPtr();
	Int* getColIndicesCacheGPUPtr();

private:

	UInt m_ColSize = 0;
	UInt m_RowSize = 0;
	UInt m_NonZeroSize = 0;
	UInt m_IterationNum = 100;
	Real m_Threshold = 1e-7;

	CCuDenseVector m_ZM1;
	CCuDenseVector m_ZM2;
	CCuDenseVector m_R;
	CCuDenseVector m_RM1;
	CCuDenseVector m_RM2;
	CCuDenseVector m_P;
	CCuDenseVector m_AP;
	CCuDenseVector m_TempVector;
	thrust::device_vector<Real> m_ILU0Value;
	thrust::device_vector<Real> m_NonZeroCache;//Sparese Matrix A
	thrust::device_vector<Int> m_RowIndicesCache;//Sparese Matrix A
	thrust::device_vector<Int> m_ColIndicesCache;//Sparese Matrix A
	Real One = 1.0;
	Real Zero = 0.0;
	void* m_PCGILU0Buffer = nullptr;
	cusparseMatDescr_t DescrA = 0;
	csrilu02Info_t InfoILU = NULL;
	cusparseMatDescr_t DescrL = NULL;
	csrsv2Info_t InfoL = NULL;
	cusparseMatDescr_t DescrU = NULL;
	csrsv2Info_t InfoU = NULL;
	cusparseSpMatDescr_t MatA = NULL;
};

