// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMLibDefines.h"
#include "Engines/Meshing/CUDA/ITMMeshingEngine_CUDA.tcu"
#include "Engines/Meshing/CUDA/ITMMultiMeshingEngine_CUDA.tcu"
#include "Engines/Reconstruction/CUDA/ITMSceneReconstructionEngine_CUDA.tcu"
#include "Engines/Reconstruction/CUDA/ITMSurfelSceneReconstructionEngine_CUDA.tcu"
#include "Engines/Swapping/CUDA/ITMSwappingEngine_CUDA.tcu"
#include "Engines/Visualisation/CUDA/ITMSurfelVisualisationEngine_CUDA.tcu"
#include "Engines/Visualisation/CUDA/ITMVisualisationEngine_CUDA.tcu"
#include "Engines/Visualisation/CUDA/ITMMultiVisualisationEngine_CUDA.tcu"

namespace ITMLib
{
	template class ITMSwappingEngine_CUDA<ITMVoxel>;

	template class ITMSurfelSceneReconstructionEngine_CUDA<ITMSurfel_grey>;
	template class ITMSurfelSceneReconstructionEngine_CUDA<ITMSurfel_rgb>;
	template class ITMSurfelVisualisationEngine_CUDA<ITMSurfel_grey>;
	template class ITMSurfelVisualisationEngine_CUDA<ITMSurfel_rgb>;
}
