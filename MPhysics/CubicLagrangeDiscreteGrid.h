#pragma once
#include "Common.h"

#define NodePerCell 32

struct SCLDGridInfo
{
	SAABB Domain;
	Vector3ui Resolution;
	Vector3 CellSize;
	Vector3 InvCellSize;
	UInt TotalCellNum = 0;
	UInt TotalFieldNum = 0;

	UInt TotalVertexNum = 0;
	UInt TotalEdgeNum = 0;
	UInt TotalNodeNum = 0;

	UInt XEdgeNum = 0;
	UInt YEdgeNum = 0;
	UInt ZEdgeNum = 0;

	__host__ __device__ Vector3 indexToNodePosition(UInt vIndex) const 
	{
		Vector3 Pos = Vector3(0, 0, 0);

		auto N = Resolution;

		auto nv = (N.x + 1) * (N.y + 1) * (N.z + 1);
		auto ne_x = (N.x + 0) * (N.y + 1) * (N.z + 1);
		auto ne_y = (N.x + 1) * (N.y + 0) * (N.z + 1);
		auto ne_z = (N.x + 1) * (N.y + 1) * (N.z + 0);
		auto ne = ne_x + ne_y + ne_z;

		auto ijk = Vector3ui(0, 0, 0);
		if (vIndex < nv)
		{
			ijk.z = vIndex / ((N.y + 1) * (N.x + 1));
			auto temp = vIndex % ((N.y + 1) * (N.x + 1));
			ijk.y = temp / (N.x + 1);
			ijk.x = temp % (N.x + 1);

			Pos = Domain.Min + CellSize * castToVector3(ijk);
		}
		else if (vIndex < nv + 2 * ne_x)
		{
			vIndex -= nv;
			auto e_ind = vIndex / 2;
			ijk.z = e_ind / ((N.y + 1) * N.x);
			auto temp = e_ind % ((N.y + 1) * N.x);
			ijk.y = temp / N.x;
			ijk.x = temp % N.x;

			Pos = Domain.Min + CellSize * castToVector3(ijk);
			Pos.x += (1.0 + static_cast<double>(vIndex % 2)) / 3.0 * CellSize.x;
		}
		else if (vIndex < nv + 2 * (ne_x + ne_y))
		{
			vIndex -= (nv + 2 * ne_x);
			auto e_ind = vIndex / 2;
			ijk.x = e_ind / ((N.z + 1) * N.y);
			auto temp = e_ind % ((N.z + 1) * N.y);
			ijk.z = temp / N.y;
			ijk.y = temp % N.y;

			Pos = Domain.Min + CellSize * castToVector3(ijk);
			Pos.y += (1.0 + static_cast<double>(vIndex % 2)) / 3.0 * CellSize.y;
		}
		else
		{
			vIndex -= (nv + 2 * (ne_x + ne_y));
			auto e_ind = vIndex / 2;
			ijk.y = e_ind / ((N.x + 1) * N.z);
			auto temp = e_ind % ((N.x + 1) * N.z);
			ijk.x = temp / N.z;
			ijk.z = temp % N.z;

			Pos = Domain.Min + CellSize * castToVector3(ijk);
			Pos.z += (1.0 + static_cast<double>(vIndex % 2)) / 3.0 * CellSize.z;
		}

		return Pos;
	}
	__host__ __device__ Vector3ui singleToMultiIndex(UInt vIndex) const
	{
		UInt n01 = Resolution.x * Resolution.y;
		UInt k = vIndex / n01;
		UInt temp = vIndex % n01;
		UInt j = temp / Resolution.x;
		UInt i = temp % Resolution.x;
		return Vector3ui(i, j, k);
	}
	__host__ __device__ UInt multiToSingleIndex(const Vector3ui& ijk) const
	{
		return Resolution.y * Resolution.x * ijk.z + Resolution.x * ijk.y + ijk.x;
	}
	__host__ __device__ SAABB subdomain(const Vector3ui& ijk) const
	{
		SAABB SubDomain;
		Vector3 Origin = Domain.Min + castToVector3(ijk) * CellSize;
		SubDomain.Min = Origin;
		SubDomain.Max = Origin + CellSize;
		return SubDomain;
	}
	__host__ __device__ SAABB subdomain(UInt vIndex) const
	{
		return subdomain(singleToMultiIndex(vIndex));
	}

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & Domain;
		ar & Resolution;
		ar & CellSize;
		ar & InvCellSize;
		ar & TotalCellNum;
		ar & TotalFieldNum;
		ar & TotalVertexNum;
		ar & TotalEdgeNum;
		ar & TotalNodeNum;
		ar & XEdgeNum;
		ar & YEdgeNum;
		ar & ZEdgeNum;
	}
};

using ContinuousFunction = std::function<void(SCLDGridInfo vCLDGridInfo, Real* vioGridNodeValue, const UInt* vCellData)>;

//交错网格，cell的各个边有两个node，格点上有一个Node，面上没有node
class CCubicLagrangeDiscreteGrid
{
public:
	CCubicLagrangeDiscreteGrid() = default;
	CCubicLagrangeDiscreteGrid(const SAABB& vDomain, const Vector3ui& vResolution, UInt vFieldNum = 1);

	void logFieldData(UInt vIndex);

	void store(const string& vPath);
	void load(const string& vPath);

	void setNodeValue(UInt FieldIndex, const ContinuousFunction& vFunc);

	void interpolateLargeDataSet
	(
		UInt FieldIndex,
		UInt XSize,
		const Real* vX,
		Real* voResult,
		Real* voGradient,
		Vector3 vGridTransform,
		SMatrix3x3 vGridRotation
	) const;

	const SAABB& getDomain() const { return m_CLDGridInfo.Domain; }
	const Vector3ui& getResolution() const { return m_CLDGridInfo.Resolution; }
	const Vector3& getCellSize() const { return m_CLDGridInfo.CellSize; }
	const Vector3& invCellSize() const { return m_CLDGridInfo.InvCellSize; }
	UInt getTotalCellNum() const { return m_CLDGridInfo.TotalCellNum; }
	UInt getTotalFieldNum() const { return m_CLDGridInfo.TotalFieldNum; }
	thrust::device_vector<Real>& getField(UInt vFieldIndex);
	thrust::device_vector<UInt>& getCell();
	SCLDGridInfo getCLDGridInfo() const;


private:
	vector<thrust::device_vector<Real>> m_Nodes; //网格的Node数+网格边上的Node数，线性排列，转换关系见__indexToNodePosition函数
	thrust::device_vector<UInt> m_Cells; //CellSize x 32，记录每个Cell上32个Node的索引
	SCLDGridInfo m_CLDGridInfo;
};