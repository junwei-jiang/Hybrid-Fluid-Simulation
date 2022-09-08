#include "CubicLagrangeDiscreteGrid.h"
#include "ThrustWapper.cuh"
#include "CLDGridKernel.cuh"

CCubicLagrangeDiscreteGrid::CCubicLagrangeDiscreteGrid(const SAABB & vDomain, const Vector3ui & vResolution, UInt vFieldNum)
{
	m_CLDGridInfo.Domain = vDomain;
	m_CLDGridInfo.Resolution = vResolution;

	Vector3 Dia = vDomain.Max - vDomain.Min;
	m_CLDGridInfo.CellSize = Dia / castToVector3(vResolution);
	m_CLDGridInfo.InvCellSize = 1.0 / m_CLDGridInfo.CellSize;
	m_CLDGridInfo.TotalCellNum = vResolution.x * vResolution.y * vResolution.z;
	Vector3ui Res = m_CLDGridInfo.Resolution;

	m_CLDGridInfo.TotalVertexNum = (Res.x + 1) * (Res.y + 1) * (Res.z + 1);
	m_CLDGridInfo.XEdgeNum = (Res.x + 0) * (Res.y + 1) * (Res.z + 1);
	m_CLDGridInfo.YEdgeNum = (Res.x + 1) * (Res.y + 0) * (Res.z + 1);
	m_CLDGridInfo.ZEdgeNum = (Res.x + 1) * (Res.y + 1) * (Res.z + 0);
	m_CLDGridInfo.TotalEdgeNum = m_CLDGridInfo.XEdgeNum + m_CLDGridInfo.YEdgeNum + m_CLDGridInfo.ZEdgeNum;
	m_CLDGridInfo.TotalNodeNum = m_CLDGridInfo.TotalVertexNum + 2 * m_CLDGridInfo.TotalEdgeNum;

	m_Nodes.resize(vFieldNum);
	m_CLDGridInfo.TotalFieldNum = vFieldNum;
	for (UInt i = 0; i < vFieldNum; i++)
	{
		resizeDeviceVector(m_Nodes[i], m_CLDGridInfo.TotalNodeNum);
	}

	resizeDeviceVector(m_Cells, m_CLDGridInfo.TotalCellNum * NodePerCell);
	fillCellRelativeNodeIndexInvoker
	(
		m_CLDGridInfo.TotalCellNum,
		m_CLDGridInfo.Resolution,
		m_CLDGridInfo.TotalVertexNum,
		m_CLDGridInfo.XEdgeNum,
		m_CLDGridInfo.YEdgeNum,
		m_CLDGridInfo.ZEdgeNum,
		getRawDevicePointerUInt(m_Cells)
	);
	
}

void CCubicLagrangeDiscreteGrid::logFieldData(UInt vIndex)
{
	Vector3ui Res = m_CLDGridInfo.Resolution;
	for (int z = 0; z < Res.z; ++z)
	{
		for (int y = 0; y < Res.y; ++y)
		{
			for (int x = 0; x < Res.x; x++)
			{

				UInt NodeIndex = getElementUInt(m_Cells, m_CLDGridInfo.multiToSingleIndex(Vector3ui(x, y, z)) * NodePerCell + 30);
				cout << getElementReal(m_Nodes[vIndex], NodeIndex) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

void CCubicLagrangeDiscreteGrid::store(const string & vPath)
{
	{
		ofstream CLDDATA_Cache(vPath + "CLDGridDATA.cache");
		boost::archive::text_oarchive oa(CLDDATA_Cache);
		oa & m_CLDGridInfo;
		CLDDATA_Cache.close();
	}

	{
		for (UInt i = 0; i < m_CLDGridInfo.TotalFieldNum; i++)
		{
			Real* FieldCPU = (Real*)malloc(sizeof(Real) * m_CLDGridInfo.TotalNodeNum);
			CHECK_CUDA(cudaMemcpy(FieldCPU, getReadOnlyRawDevicePointer(m_Nodes[i]), m_CLDGridInfo.TotalNodeNum * sizeof(Real), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaDeviceSynchronize());

			FILE * FilePointer = fopen((vPath + "NodeDATA_" + to_string(i) + ".cache").c_str(), "w+b");
			UInt DataWrite = fwrite(FieldCPU, sizeof(Real), m_CLDGridInfo.TotalNodeNum, FilePointer);
			fclose(FilePointer);

			free(FieldCPU);
		}
	}
}

void CCubicLagrangeDiscreteGrid::load(const string & vPath)
{
	{
		ifstream CLDDATA_Cache(vPath + "CLDGridDATA.cache");
		boost::archive::text_iarchive ia(CLDDATA_Cache);
		ia & m_CLDGridInfo;
		CLDDATA_Cache.close();
	}

	m_Nodes.resize(m_CLDGridInfo.TotalFieldNum);
	for (int i = 0; i < m_CLDGridInfo.TotalFieldNum; i++)
	{
		resizeDeviceVector(m_Nodes[i], m_CLDGridInfo.TotalNodeNum);
	}

	{
		for (UInt i = 0; i < m_CLDGridInfo.TotalFieldNum; i++)
		{
			Real* FieldCPU = (Real*)malloc(sizeof(Real) * m_CLDGridInfo.TotalNodeNum);
			FILE * FilePointer = fopen((vPath + "NodeDATA_" + to_string(i) + ".cache").c_str(), "rb");
			UInt DataRead = fread(FieldCPU, sizeof(Real), m_CLDGridInfo.TotalNodeNum, FilePointer);
			fclose(FilePointer);
			
			CHECK_CUDA(cudaMemcpy(getRawDevicePointerReal(m_Nodes[i]), FieldCPU, m_CLDGridInfo.TotalNodeNum * sizeof(Real), cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaDeviceSynchronize());
			free(FieldCPU);
		}
	}

	resizeDeviceVector(m_Cells, m_CLDGridInfo.TotalCellNum * NodePerCell);
	fillCellRelativeNodeIndexInvoker
	(
		m_CLDGridInfo.TotalCellNum,
		m_CLDGridInfo.Resolution,
		m_CLDGridInfo.TotalVertexNum,
		m_CLDGridInfo.XEdgeNum,
		m_CLDGridInfo.YEdgeNum,
		m_CLDGridInfo.ZEdgeNum,
		getRawDevicePointerUInt(m_Cells)
	);
}

void CCubicLagrangeDiscreteGrid::setNodeValue(UInt FieldIndex, const ContinuousFunction & vFunc)
{
	vFunc(m_CLDGridInfo, getRawDevicePointerReal(m_Nodes[FieldIndex]), getReadOnlyRawDevicePointer(m_Cells));
}

void CCubicLagrangeDiscreteGrid::interpolateLargeDataSet
(
	UInt FieldIndex,
	UInt XSize,
	const Real* vX, 
	Real* voResult,
	Real* voGradient,
	Vector3 vGridTransform,
	SMatrix3x3 vGridRotation
) const
{
	interpolateInvoker
	(
		vX,
		XSize,
		getReadOnlyRawDevicePointer(m_Nodes[FieldIndex]),
		getReadOnlyRawDevicePointer(m_Cells),
		m_CLDGridInfo,
		voResult,
		voGradient,
		vGridTransform,
		vGridRotation
	);
}

thrust::device_vector<Real>& CCubicLagrangeDiscreteGrid::getField(UInt vFieldIndex)
{
	return m_Nodes[vFieldIndex];
}

thrust::device_vector<UInt>& CCubicLagrangeDiscreteGrid::getCell()
{
	return m_Cells;
}

SCLDGridInfo CCubicLagrangeDiscreteGrid::getCLDGridInfo() const
{
	return m_CLDGridInfo;
}
