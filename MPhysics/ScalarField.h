#pragma once
#include "Field.h"

class CCellCenteredScalarField;
class CCellCenteredVectorField;

class CScalarField : public CField
{
public:
	CScalarField();
	virtual ~CScalarField();

	virtual void gradient(CCellCenteredVectorField& voGradientField) const;
	virtual void laplacian(CCellCenteredScalarField& voLaplacianField) const;
};
