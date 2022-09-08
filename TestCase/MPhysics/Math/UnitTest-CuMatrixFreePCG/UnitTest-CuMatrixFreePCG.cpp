#include "pch.h"
#include "CuSparseMatrix.h"
#include "CuDenseVector.h"
#include "CuMatrixFreePCG.h"
#include "CuMatVecMultiplier.h"
#include "CudaContextManager.h"

struct SUserData
{
	unique_ptr<CCuSparseMatrix> A;
	unique_ptr<CCuSparseMatrix> MInv;
};

static void TestAxCGProd(const CCuDenseVector& vVector, CCuDenseVector& voResult, void* vUserMatrix)
{
	SUserData* Data = (SUserData*)vUserMatrix;
	CCuMatVecMultiplier CuSparseMV = CCuMatVecMultiplier();
	CuSparseMV.mulMatVec(*Data->A, vVector, voResult);
}

static void TestMInvxCGProd(const CCuDenseVector& vVector, CCuDenseVector& voResult, void* vUserMatrix)
{
	SUserData* Data = (SUserData*)vUserMatrix;
	CCuMatVecMultiplier CuSparseMV = CCuMatVecMultiplier();
	CuSparseMV.mulMatVec(*Data->MInv, vVector, voResult);
}


TEST(CuMatrixFreePCG, CuMatrixFreePCG_Main)
{
	CCudaContextManager::getInstance().initCudaContext();

	SUserData UserData;

	Real* NonZeroA = new Real[9]{ 4,-2,-1,-2,4,-2,-1,-2,3 };
	Int* RowIndexA = new Int[9]{ 0,0,0,1,1,1,2,2,2 };
	Int* ColIndexA = new Int[9]{ 0,1,2,0,1,2,0,1,2 };
	UserData.A = unique_ptr<CCuSparseMatrix>(new CCuSparseMatrix(NonZeroA, RowIndexA, ColIndexA, 9, 3, 3));

	Real* NonZeroM = new Real[3]{ 1.0/4.0, 1.0/4.0, 1.0/3.0 };
	//Real* NonZeroM = new Real[3]{ 1.0, 1.0, 1.0 };
	Int* RowIndexM = new Int[3]{ 0, 1, 2 };
	Int* ColIndexM = new Int[3]{ 0, 1, 2 };
	UserData.MInv = unique_ptr<CCuSparseMatrix>(new CCuSparseMatrix(NonZeroM, RowIndexM, ColIndexM, 3, 3, 3));

	Real* Vectorb = new Real[3]{ 0, -2, 3 };
	CCuDenseVector VectorbGPU = CCuDenseVector(Vectorb, 3);
	CCuDenseVector X = CCuDenseVector(3);

	CCuMatrixFreePCG PCGsolver;
	PCGsolver.init(3, 3, 3, TestAxCGProd, TestMInvxCGProd);
	PCGsolver.setIterationNum(1000);
	PCGsolver.solvePCGInvDia(VectorbGPU, X, &UserData);

	CCuDenseVector VO(3);
	CCuMatVecMultiplier CuSparseMV;
	CuSparseMV.mulMatVec(*UserData.A, X, VO);
	VO *= -1;

	CCuDenseVector Error(VectorbGPU);
	Error += VO;
	Real RelativeError = abs(Error.norm2());
	EXPECT_LE(RelativeError, 1e-6);

	free(NonZeroA);
	free(RowIndexA);
	free(ColIndexA);

	free(NonZeroM);
	free(RowIndexM);
	free(ColIndexM);

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuMatrixFreePCG, CuMatrixFreePCG_Main2)
{
	CCudaContextManager::getInstance().initCudaContext();

	SUserData UserData;

	Real* NonZeroA = new Real[32]{ 3,-1,-1,-1,-1,3,-1,-1,-1,3,-1,-1,-1,-1,3,-1,-1,3,-1,-1,-1,-1,3,-1,-1,-1,3,-1,-1,-1,-1,1 };
	Int* RowIndexA = new Int[32]{ 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7 };
	Int* ColIndexA = new Int[32]{ 0,1,2,4,0,1,3,5,0,2,3,6,1,2,3,7,0,4,5,6,1,4,5,7,2,4,6,7,3,5,6,7 };
	UserData.A = unique_ptr<CCuSparseMatrix>(new CCuSparseMatrix(NonZeroA, RowIndexA, ColIndexA, 32, 8, 8));

	//Real* NonZeroM = new Real[8]{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	Real* NonZeroM = new Real[8]{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 };
	Int* RowIndexM = new Int[8]{ 0, 1, 2, 3, 4, 5, 6, 7};
	Int* ColIndexM = new Int[8]{ 0, 1, 2, 3, 4, 5, 6, 7 };
	UserData.MInv = unique_ptr<CCuSparseMatrix>(new CCuSparseMatrix(NonZeroM, RowIndexM, ColIndexM, 8, 8, 8));

	Real* Vectorb = new Real[8]{ -8, -16, -55, -28, -110, 52, 55, -33 };
	CCuDenseVector VectorbGPU = CCuDenseVector(Vectorb, 8);
	CCuDenseVector X = CCuDenseVector(8);

	CCuMatrixFreePCG PCGsolver;
	PCGsolver.init(8, 8, 8, TestAxCGProd, TestMInvxCGProd);
	PCGsolver.setIterationNum(8);

	for (int i = 0; i < 500; i++)
	{
		PCGsolver.solvePCGInvDia(VectorbGPU, X, &UserData);

		CCuDenseVector VO(8);
		CCuMatVecMultiplier CuSparseMV;
		CuSparseMV.mulMatVec(*UserData.A, X, VO);
		VO *= -1;

		CCuDenseVector Error(VectorbGPU);
		Error += VO;
		Real RelativeError = abs(Error.norm2());
		EXPECT_LE(RelativeError, 1e-4);
	}

	free(NonZeroA);
	free(RowIndexA);
	free(ColIndexA);

	free(NonZeroM);
	free(RowIndexM);
	free(ColIndexM);

	CCudaContextManager::getInstance().freeCudaContext();
}