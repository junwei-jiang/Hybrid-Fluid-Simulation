﻿#include"GaussQuadrature.cuh"

__constant__ __device__ Real gaussian_abscissae_1_30[16] = {
	-0.989400934991649938510249739920,
	-0.944575023073232600268056557979,
	-0.865631202387831755196145877562,
	-0.755404408355002998654015300417,
	-0.617876244402643770570193737512,
	-0.458016777657227369680015272024,
	-0.281603550779258915426339626720,
	-0.095012509837637426635126303154,
	0.095012509837637426635126303154,
	0.281603550779258915426339626720,
	0.458016777657227369680015272024,
	0.617876244402643770570193737512,
	0.755404408355002998654015300417,
	0.865631202387831755196145877562,
	0.944575023073232600268056557979,
	0.989400934991649938510249739920
};

__constant__ __device__ Real gaussian_weights_1_30[16] = {
	0.027152459411758110563450685504,
	0.062253523938649010793788818319,
	0.095158511682492036287683845330,
	0.124628971255533488315947465708,
	0.149595988816575764523975067277,
	0.169156519395001675443168664970,
	0.182603415044922529064663763165,
	0.189450610455067447457366824892,
	0.189450610455067447457366824892,
	0.182603415044922529064663763165,
	0.169156519395001675443168664970,
	0.149595988816575764523975067277,
	0.124628971255533488315947465708,
	0.095158511682492036287683845330,
	0.062253523938649010793788818319,
	0.027152459411758110563450685504
};