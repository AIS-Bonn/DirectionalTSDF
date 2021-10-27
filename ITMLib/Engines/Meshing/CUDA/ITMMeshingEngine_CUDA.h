// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMMeshingEngine.h"

namespace ITMLib
{
class ITMMeshingEngine_CUDA : public ITMMeshingEngine
{
private:
	unsigned int* noTriangles_device{};

	/**
	 * Generate entire mesh on GPU
	 * @param mesh
	 * @param scene
	 */
	void MeshSceneDefault(ITMMesh* mesh, const Scene* scene);

	/**
	 * Alternate between meshing and copying triangles to reduce amount of simultaneously required GPU memory
	 * @param mesh
	 * @param scene
	 */
	void MeshSceneStreamed(ITMMesh* mesh, const Scene* scene);

public:
	void MeshScene(ITMMesh* mesh, const Scene* scene);

	ITMMeshingEngine_CUDA(void);

	~ITMMeshingEngine_CUDA(void);
};
}
