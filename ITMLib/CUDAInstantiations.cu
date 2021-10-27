// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "Engines/Meshing/CUDA/ITMMeshingEngine_CUDA.tcu"
#include "Engines/Visualisation/CUDA/ITMVisualisationEngine_CUDA.tcu"
#include "Engines/Reconstruction/CUDA/ITMSceneReconstructionEngine_CUDA.tcu"

namespace ITMLib
{
template
class ITMSceneReconstructionEngine_CUDA<ITMIndex>;

template
class ITMSceneReconstructionEngine_CUDA<ITMIndexDirectional>;
}
