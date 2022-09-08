#pragma once
#include "CLDGridinterpolate.cuh"

__host__ __device__ void genShapeFunction(const Vector3& vXi, Vector3 voDN[32], Real voN[32])
{
	Real x = vXi.x;
	Real y = vXi.y;
	Real z = vXi.z;

	Real x2 = x * x;
	Real y2 = y * y;
	Real z2 = z * z;

	Real _1mx = 1.0 - x;
	Real _1my = 1.0 - y;
	Real _1mz = 1.0 - z;

	Real _1px = 1.0 + x;
	Real _1py = 1.0 + y;
	Real _1pz = 1.0 + z;

	Real _1m3x = 1.0 - 3.0 * x;
	Real _1m3y = 1.0 - 3.0 * y;
	Real _1m3z = 1.0 - 3.0 * z;

	Real _1p3x = 1.0 + 3.0 * x;
	Real _1p3y = 1.0 + 3.0 * y;
	Real _1p3z = 1.0 + 3.0 * z;

	Real _1mxt1my = _1mx * _1my;
	Real _1mxt1py = _1mx * _1py;
	Real _1pxt1my = _1px * _1my;
	Real _1pxt1py = _1px * _1py;

	Real _1mxt1mz = _1mx * _1mz;
	Real _1mxt1pz = _1mx * _1pz;
	Real _1pxt1mz = _1px * _1mz;
	Real _1pxt1pz = _1px * _1pz;

	Real _1myt1mz = _1my * _1mz;
	Real _1myt1pz = _1my * _1pz;
	Real _1pyt1mz = _1py * _1mz;
	Real _1pyt1pz = _1py * _1pz;

	Real _1mx2 = 1.0 - x2;
	Real _1my2 = 1.0 - y2;
	Real _1mz2 = 1.0 - z2;

	// Corner nodes.
	Real fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
	voN[0] = fac * _1mxt1my * _1mz;
	voN[1] = fac * _1pxt1my * _1mz;
	voN[2] = fac * _1mxt1py * _1mz;
	voN[3] = fac * _1pxt1py * _1mz;
	voN[4] = fac * _1mxt1my * _1pz;
	voN[5] = fac * _1pxt1my * _1pz;
	voN[6] = fac * _1mxt1py * _1pz;
	voN[7] = fac * _1pxt1py * _1pz;

	// Edge nodes.

	fac = 9.0 / 64.0 * _1mx2;
	Real fact1m3x = fac * _1m3x;
	Real fact1p3x = fac * _1p3x;
	voN[8] = fact1m3x * _1myt1mz;
	voN[9] = fact1p3x * _1myt1mz;
	voN[10] = fact1m3x * _1myt1pz;
	voN[11] = fact1p3x * _1myt1pz;
	voN[12] = fact1m3x * _1pyt1mz;
	voN[13] = fact1p3x * _1pyt1mz;
	voN[14] = fact1m3x * _1pyt1pz;
	voN[15] = fact1p3x * _1pyt1pz;

	fac = 9.0 / 64.0 * _1my2;
	Real fact1m3y = fac * _1m3y;
	Real fact1p3y = fac * _1p3y;
	voN[16] = fact1m3y * _1mxt1mz;
	voN[17] = fact1p3y * _1mxt1mz;
	voN[18] = fact1m3y * _1pxt1mz;
	voN[19] = fact1p3y * _1pxt1mz;
	voN[20] = fact1m3y * _1mxt1pz;
	voN[21] = fact1p3y * _1mxt1pz;
	voN[22] = fact1m3y * _1pxt1pz;
	voN[23] = fact1p3y * _1pxt1pz;

	fac = 9.0 / 64.0 * _1mz2;
	Real fact1m3z = fac * _1m3z;
	Real fact1p3z = fac * _1p3z;
	voN[24] = fact1m3z * _1mxt1my;
	voN[25] = fact1p3z * _1mxt1my;
	voN[26] = fact1m3z * _1mxt1py;
	voN[27] = fact1p3z * _1mxt1py;
	voN[28] = fact1m3z * _1pxt1my;
	voN[29] = fact1p3z * _1pxt1my;
	voN[30] = fact1m3z * _1pxt1py;
	voN[31] = fact1p3z * _1pxt1py;

	if (voDN)
	{
		Real _9t3x2py2pz2m19 = 9.0 * (3.0 * x2 + y2 + z2) - 19.0;
		Real _9tx2p3y2pz2m19 = 9.0 * (x2 + 3.0 * y2 + z2) - 19.0;
		Real _9tx2py2p3z2m19 = 9.0 * (x2 + y2 + 3.0 * z2) - 19.0;
		Real _18x = 18.0 * x;
		Real _18y = 18.0 * y;
		Real _18z = 18.0 * z;

		Real _3m9x2 = 3.0 - 9.0 * x2;
		Real _3m9y2 = 3.0 - 9.0 * y2;
		Real _3m9z2 = 3.0 - 9.0 * z2;

		Real _2x = 2.0 * x;
		Real _2y = 2.0 * y;
		Real _2z = 2.0 * z;

		Real _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
		Real _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
		Real _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
		Real _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
		Real _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
		Real _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

		voDN[0].x = _18xm9t3x2py2pz2m19 * _1myt1mz;
		voDN[0].y = _1mxt1mz * _18ym9tx2p3y2pz2m19;
		voDN[0].z = _1mxt1my * _18zm9tx2py2p3z2m19;
		voDN[1].x = _18xp9t3x2py2pz2m19 * _1myt1mz;
		voDN[1].y = _1pxt1mz * _18ym9tx2p3y2pz2m19;
		voDN[1].z = _1pxt1my * _18zm9tx2py2p3z2m19;
		voDN[2].x = _18xm9t3x2py2pz2m19 * _1pyt1mz;
		voDN[2].y = _1mxt1mz * _18yp9tx2p3y2pz2m19;
		voDN[2].z = _1mxt1py * _18zm9tx2py2p3z2m19;
		voDN[3].x = _18xp9t3x2py2pz2m19 * _1pyt1mz;
		voDN[3].y = _1pxt1mz * _18yp9tx2p3y2pz2m19;
		voDN[3].z = _1pxt1py * _18zm9tx2py2p3z2m19;
		voDN[4].x = _18xm9t3x2py2pz2m19 * _1myt1pz;
		voDN[4].y = _1mxt1pz * _18ym9tx2p3y2pz2m19;
		voDN[4].z = _1mxt1my * _18zp9tx2py2p3z2m19;
		voDN[5].x = _18xp9t3x2py2pz2m19 * _1myt1pz;
		voDN[5].y = _1pxt1pz * _18ym9tx2p3y2pz2m19;
		voDN[5].z = _1pxt1my * _18zp9tx2py2p3z2m19;
		voDN[6].x = _18xm9t3x2py2pz2m19 * _1pyt1pz;
		voDN[6].y = _1mxt1pz * _18yp9tx2p3y2pz2m19;
		voDN[6].z = _1mxt1py * _18zp9tx2py2p3z2m19;
		voDN[7].x = _18xp9t3x2py2pz2m19 * _1pyt1pz;
		voDN[7].y = _1pxt1pz * _18yp9tx2p3y2pz2m19;
		voDN[7].z = _1pxt1py * _18zp9tx2py2p3z2m19;

		for (int i = 0; i < 8; i++) voDN[i] /= 64.0;

		Real _m3m9x2m2x = -_3m9x2 - _2x;
		Real _p3m9x2m2x = _3m9x2 - _2x;
		Real _1mx2t1m3x = _1mx2 * _1m3x;
		Real _1mx2t1p3x = _1mx2 * _1p3x;
		voDN[8].x = _m3m9x2m2x * _1myt1mz;
		voDN[8].y = -_1mx2t1m3x * _1mz;
		voDN[8].z = -_1mx2t1m3x * _1my;
		voDN[9].x = _p3m9x2m2x * _1myt1mz;
		voDN[9].y = -_1mx2t1p3x * _1mz;
		voDN[9].z = -_1mx2t1p3x * _1my;
		voDN[10].x = _m3m9x2m2x * _1myt1pz;
		voDN[10].y = -_1mx2t1m3x * _1pz;
		voDN[10].z = _1mx2t1m3x * _1my;
		voDN[11].x = _p3m9x2m2x * _1myt1pz;
		voDN[11].y = -_1mx2t1p3x * _1pz;
		voDN[11].z = _1mx2t1p3x * _1my;
		voDN[12].x = _m3m9x2m2x * _1pyt1mz;
		voDN[12].y = _1mx2t1m3x * _1mz;
		voDN[12].z = -_1mx2t1m3x * _1py;
		voDN[13].x = _p3m9x2m2x * _1pyt1mz;
		voDN[13].y = _1mx2t1p3x * _1mz;
		voDN[13].z = -_1mx2t1p3x * _1py;
		voDN[14].x = _m3m9x2m2x * _1pyt1pz;
		voDN[14].y = _1mx2t1m3x * _1pz;
		voDN[14].z = _1mx2t1m3x * _1py;
		voDN[15].x = _p3m9x2m2x * _1pyt1pz;
		voDN[15].y = _1mx2t1p3x * _1pz;
		voDN[15].z = _1mx2t1p3x * _1py;

		Real _m3m9y2m2y = -_3m9y2 - _2y;
		Real _p3m9y2m2y = _3m9y2 - _2y;
		Real _1my2t1m3y = _1my2 * _1m3y;
		Real _1my2t1p3y = _1my2 * _1p3y;
		voDN[16].x = -_1my2t1m3y * _1mz;
		voDN[16].y = _m3m9y2m2y * _1mxt1mz;
		voDN[16].z = -_1my2t1m3y * _1mx;
		voDN[17].x = -_1my2t1p3y * _1mz;
		voDN[17].y = _p3m9y2m2y * _1mxt1mz;
		voDN[17].z = -_1my2t1p3y * _1mx;
		voDN[18].x = _1my2t1m3y * _1mz;
		voDN[18].y = _m3m9y2m2y * _1pxt1mz;
		voDN[18].z = -_1my2t1m3y * _1px;
		voDN[19].x = _1my2t1p3y * _1mz;
		voDN[19].y = _p3m9y2m2y * _1pxt1mz;
		voDN[19].z = -_1my2t1p3y * _1px;
		voDN[20].x = -_1my2t1m3y * _1pz;
		voDN[20].y = _m3m9y2m2y * _1mxt1pz;
		voDN[20].z = _1my2t1m3y * _1mx;
		voDN[21].x = -_1my2t1p3y * _1pz;
		voDN[21].y = _p3m9y2m2y * _1mxt1pz;
		voDN[21].z = _1my2t1p3y * _1mx;
		voDN[22].x = _1my2t1m3y * _1pz;
		voDN[22].y = _m3m9y2m2y * _1pxt1pz;
		voDN[22].z = _1my2t1m3y * _1px;
		voDN[23].x = _1my2t1p3y * _1pz;
		voDN[23].y = _p3m9y2m2y * _1pxt1pz;
		voDN[23].z = _1my2t1p3y * _1px;

		Real _m3m9z2m2z = -_3m9z2 - _2z;
		Real _p3m9z2m2z = _3m9z2 - _2z;
		Real _1mz2t1m3z = _1mz2 * _1m3z;
		Real _1mz2t1p3z = _1mz2 * _1p3z;
		voDN[24].x = -_1mz2t1m3z * _1my;
		voDN[24].y = -_1mz2t1m3z * _1mx;
		voDN[24].z = _m3m9z2m2z * _1mxt1my;
		voDN[25].x = -_1mz2t1p3z * _1my;
		voDN[25].y = -_1mz2t1p3z * _1mx;
		voDN[25].z = _p3m9z2m2z * _1mxt1my;
		voDN[26].x = -_1mz2t1m3z * _1py;
		voDN[26].y = _1mz2t1m3z * _1mx;
		voDN[26].z = _m3m9z2m2z * _1mxt1py;
		voDN[27].x = -_1mz2t1p3z * _1py;
		voDN[27].y = _1mz2t1p3z * _1mx;
		voDN[27].z = _p3m9z2m2z * _1mxt1py;
		voDN[28].x = _1mz2t1m3z * _1my;
		voDN[28].y = -_1mz2t1m3z * _1px;
		voDN[28].z = _m3m9z2m2z * _1pxt1my;
		voDN[29].x = _1mz2t1p3z * _1my;
		voDN[29].y = -_1mz2t1p3z * _1px;
		voDN[29].z = _p3m9z2m2z * _1pxt1my;
		voDN[30].x = _1mz2t1m3z * _1py;
		voDN[30].y = _1mz2t1m3z * _1px;
		voDN[30].z = _m3m9z2m2z * _1pxt1py;
		voDN[31].x = _1mz2t1p3z * _1py;
		voDN[31].y = _1mz2t1p3z * _1px;
		voDN[31].z = _p3m9z2m2z * _1pxt1py;

		for (int i = 8u; i < 32u; i++) voDN[i] *= 9.0 / 64.0;
	}
}

__host__ __device__ void interpolateSinglePos
(
	const Vector3& vX,
	const Real* vNodeGPUPtr,
	const UInt* vCellGPUPtr,
	SCLDGridInfo vCLDGridInfo,

	Real& voValueInte,
	Vector3* voGradInte
)
{
	SAABB Domain = vCLDGridInfo.Domain;
	if (!Domain.contains(vX))
	{
		voValueInte = REAL_MAX;
		voGradInte = nullptr;
		return;
	}
	Vector3ui CellIndex = castToVector3ui((vX - Domain.Min) / vCLDGridInfo.CellSize);
	Vector3ui Res = vCLDGridInfo.Resolution;
	if (CellIndex.x >= Res.x) CellIndex.x = Res.x - 1;
	if (CellIndex.y >= Res.y) CellIndex.y = Res.y - 1;
	if (CellIndex.z >= Res.z) CellIndex.z = Res.z - 1;

	UInt CellLinerIndex = vCLDGridInfo.multiToSingleIndex(CellIndex);
	SAABB SubDomain = vCLDGridInfo.subdomain(CellIndex);
	Vector3 SBMin = SubDomain.Min;
	Vector3 SBMax = SubDomain.Max;
	Vector3 Dia = SubDomain.Max - SubDomain.Min;
	Vector3 C0 = 2.0 / Dia;
	Vector3 C1 = (SubDomain.Max + SubDomain.Min) / Dia;
	Vector3 Xi = C0 * vX - C1;

	Real N[NodePerCell];

	if (!voGradInte)
	{
		Real Phi = 0.0;
		genShapeFunction(Xi, nullptr, N);
		for (UInt i = 0; i < NodePerCell; i++)
		{
			UInt NodeIndex = vCellGPUPtr[CellLinerIndex * NodePerCell + i];
			Real NodeData = vNodeGPUPtr[NodeIndex];
			if (NodeData == REAL_MAX)
			{
				voValueInte = REAL_MAX;
				return;
			}
			Phi += NodeData * N[i];
		}
		voValueInte = Phi;
		return;
	}

	Vector3 GradN[NodePerCell];
	genShapeFunction(Xi, GradN, N);

	Real Phi = 0.0;
	Vector3 Grad = Vector3(0, 0, 0);
	for (UInt i = 0; i < NodePerCell; i++)
	{
		UInt NodeIndex = vCellGPUPtr[CellLinerIndex * NodePerCell + i];
		Real NodeData = vNodeGPUPtr[NodeIndex];
		if (NodeData == REAL_MAX)
		{
			voValueInte = REAL_MAX;
			voGradInte = nullptr;
			return;
		}
		Real Temp = N[i];
		Vector3 TempGrad = GradN[i];
		Phi += NodeData * N[i];
		Grad.x += NodeData * GradN[i].x;
		Grad.y += NodeData * GradN[i].y;
		Grad.z += NodeData * GradN[i].z;
	}
	Grad *= C0;

	voValueInte = Phi;
	*voGradInte = Grad;
}