#pragma once

enum class ESamplingAlgorithm
{
	TRILINEAR,
	CATMULLROM,
	MONOCATMULLROM,
	CUBICBRIDSON,
	CLAMPCUBICBRIDSON
};

enum class EAdvectionAccuracy
{
	RK1,
	RK2,
	RK3
};

enum class EPGTransferAlgorithm
{
	P2GSUM,
	G2PNEAREST,
	LINEAR,
	QUADRATIC,
	CUBIC
};