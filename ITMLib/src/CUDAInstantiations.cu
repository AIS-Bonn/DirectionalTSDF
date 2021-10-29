// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

/**
 * separation of header and implementation required for CUDA due to compiler incompatibility.
 * This file includes and instantiates the templated classes
 */

#include "Engines/Reconstruction/CUDA/ITMSceneReconstructionEngine_CUDA.tcu"

namespace ITMLib
{
template class ITMSceneReconstructionEngine_CUDA<ITMIndex>;
template class ITMSceneReconstructionEngine_CUDA<ITMIndexDirectional>;
}