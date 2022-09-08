#include "CLDGridKernel.cuh"
#include "CudaContextManager.h"

__global__ void fillCellRelativeNodeIndex
(
	UInt vTotalCellNum,
	Vector3ui vRes,
	UInt VertexNum,
	UInt vXEdgeNum,
	UInt vYEdgeNum,
	UInt vZEdgeNum,

	UInt* voCellNodeIndex
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vTotalCellNum) return;

	UInt k = Index / (vRes.y * vRes.x);
	UInt temp = Index % (vRes.y * vRes.x);
	UInt j = temp / vRes.x;
	UInt i = temp % vRes.x;

	UInt nx = vRes.x;
	UInt ny = vRes.y;
	UInt nz = vRes.z;

	voCellNodeIndex[Index*NodePerCell+0] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i;
	voCellNodeIndex[Index*NodePerCell+1] = (nx + 1) * (ny + 1) * k + (nx + 1) * j + i + 1;
	voCellNodeIndex[Index*NodePerCell+2] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i;
	voCellNodeIndex[Index*NodePerCell+3] = (nx + 1) * (ny + 1) * k + (nx + 1) * (j + 1) + i + 1;
	voCellNodeIndex[Index*NodePerCell+4] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i;
	voCellNodeIndex[Index*NodePerCell+5] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * j + i + 1;
	voCellNodeIndex[Index*NodePerCell+6] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i;
	voCellNodeIndex[Index*NodePerCell+7] = (nx + 1) * (ny + 1) * (k + 1) + (nx + 1) * (j + 1) + i + 1;

	UInt Offset = VertexNum;
	voCellNodeIndex[Index*NodePerCell+8] = Offset + 2 * (nx * (ny + 1) * k + nx * j + i);
	voCellNodeIndex[Index*NodePerCell+9] = voCellNodeIndex[Index*NodePerCell+8] + 1;
	voCellNodeIndex[Index*NodePerCell+10] = Offset + 2 * (nx * (ny + 1) * (k + 1) + nx * j + i);
	voCellNodeIndex[Index*NodePerCell+11] = voCellNodeIndex[Index*NodePerCell+10] + 1;
	voCellNodeIndex[Index*NodePerCell+12] = Offset + 2 * (nx * (ny + 1) * k + nx * (j + 1) + i);
	voCellNodeIndex[Index*NodePerCell+13] = voCellNodeIndex[Index*NodePerCell+12] + 1;
	voCellNodeIndex[Index*NodePerCell+14] = Offset + 2 * (nx * (ny + 1) * (k + 1) + nx * (j + 1) + i);
	voCellNodeIndex[Index*NodePerCell+15] = voCellNodeIndex[Index*NodePerCell+14] + 1;

	Offset += 2 * vXEdgeNum;
	voCellNodeIndex[Index*NodePerCell+16] = Offset + 2 * (ny * (nz + 1) * i + ny * k + j);
	voCellNodeIndex[Index*NodePerCell+17] = voCellNodeIndex[Index*NodePerCell+16] + 1;
	voCellNodeIndex[Index*NodePerCell+18] = Offset + 2 * (ny * (nz + 1) * (i + 1) + ny * k + j);
	voCellNodeIndex[Index*NodePerCell+19] = voCellNodeIndex[Index*NodePerCell+18] + 1;
	voCellNodeIndex[Index*NodePerCell+20] = Offset + 2 * (ny * (nz + 1) * i + ny * (k + 1) + j);
	voCellNodeIndex[Index*NodePerCell+21] = voCellNodeIndex[Index*NodePerCell+20] + 1;
	voCellNodeIndex[Index*NodePerCell+22] = Offset + 2 * (ny * (nz + 1) * (i + 1) + ny * (k + 1) + j);
	voCellNodeIndex[Index*NodePerCell+23] = voCellNodeIndex[Index*NodePerCell+22] + 1;

	Offset += 2 * vYEdgeNum;
	voCellNodeIndex[Index*NodePerCell+24] = Offset + 2 * (nz * (nx + 1) * j + nz * i + k);
	voCellNodeIndex[Index*NodePerCell+25] = voCellNodeIndex[Index*NodePerCell+24] + 1;
	voCellNodeIndex[Index*NodePerCell+26] = Offset + 2 * (nz * (nx + 1) * (j + 1) + nz * i + k);
	voCellNodeIndex[Index*NodePerCell+27] = voCellNodeIndex[Index*NodePerCell+26] + 1;
	voCellNodeIndex[Index*NodePerCell+28] = Offset + 2 * (nz * (nx + 1) * j + nz * (i + 1) + k);
	voCellNodeIndex[Index*NodePerCell+29] = voCellNodeIndex[Index*NodePerCell+28] + 1;
	voCellNodeIndex[Index*NodePerCell+30] = Offset + 2 * (nz * (nx + 1) * (j + 1) + nz * (i + 1) + k);
	voCellNodeIndex[Index*NodePerCell+31] = voCellNodeIndex[Index*NodePerCell+30] + 1;
}

void fillCellRelativeNodeIndexInvoker
(
	UInt vTotalCellNum,
	Vector3ui vRes,
	UInt VertexNum,
	UInt vXEdgeNum,
	UInt vYEdgeNum,
	UInt vZEdgeNum,

	UInt* voCellNodeIndex
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vTotalCellNum, BlockSize, GridSize);

	fillCellRelativeNodeIndex<<<GridSize, BlockSize>>>(vTotalCellNum, vRes, VertexNum, vXEdgeNum, vYEdgeNum, vZEdgeNum, voCellNodeIndex);
}