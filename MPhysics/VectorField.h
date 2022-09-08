#pragma once
#include "Field.h"

class CCellCenteredScalarField;
class CCellCenteredVectorField;

class CVectorField : public CField
{
public:
	CVectorField();
	virtual ~CVectorField();

	virtual void divergence(CCellCenteredScalarField& voDivergenceField) const;
	virtual void curl(CCellCenteredVectorField& voCurlField) const;
};