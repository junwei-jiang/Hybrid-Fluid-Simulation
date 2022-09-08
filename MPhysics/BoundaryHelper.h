#pragma once
#include "Common.h"
#include "CellCenteredScalarField.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class CCellCenteredScalarField;

void processMeshNode
(
	const aiNode * vNode,
	const aiScene * vSceneObjPtr,
	vector<float>& voVertex,
	vector<float>& voNormal,
	UInt& voTriangleCount
);

void generateSDF(string vTriangleMeshFilePath, CCellCenteredScalarField& voSDFField, bool vIsInvSign = false);
