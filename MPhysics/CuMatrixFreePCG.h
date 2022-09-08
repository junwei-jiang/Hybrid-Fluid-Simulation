#pragma once
#include "Common.h"
#include "CuDenseVector.h"

typedef void(*MatrixVecCGProd)(const CCuDenseVector& vVector, CCuDenseVector& voResult, void* vUserMatrix);

class CCuMatrixFreePCG
{
public:
	CCuMatrixFreePCG() = default;
	~CCuMatrixFreePCG() = default;
	void init(UInt vColSize, UInt vRowSize, UInt vNonZeroSize, MatrixVecCGProd vMatrixVecCGProd, MatrixVecCGProd vMInvxCGProdFunc);
	bool solvePCGInvDia(const CCuDenseVector& vVectorb, CCuDenseVector& voX, void* vUserData);
	bool solveCG(const CCuDenseVector & vVectorb, CCuDenseVector & voX, void* vUserData);

	void setIterationNum(UInt vIterationNum);
	void setThreshold(Real vThreshold);

	Real getThreshold();

private:
	MatrixVecCGProd m_AxCGProdFunc = nullptr;
	MatrixVecCGProd m_MInvxCGProdFunc = nullptr;
	UInt m_ColSize = 0;
	UInt m_RowSize = 0;
	UInt m_NonZeroSize = 0;
	UInt m_IterationNum = 100;
	Real m_Threshold = 1e-3;

	//CG cache
	CCuDenseVector m_ZM1;
	CCuDenseVector m_R;
	CCuDenseVector m_P;
	CCuDenseVector m_AP;
	CCuDenseVector m_TempVector;
};