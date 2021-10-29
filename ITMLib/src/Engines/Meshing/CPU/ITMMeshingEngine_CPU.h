// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Engines/ITMMeshingEngine.h>

namespace ITMLib
{
class ITMMeshingEngine_CPU : public ITMMeshingEngine
{
public:
	void MeshScene(ITMMesh* mesh, const Scene* scene) override;

	ITMMeshingEngine_CPU() = default;

	~ITMMeshingEngine_CPU() override = default;
};
}