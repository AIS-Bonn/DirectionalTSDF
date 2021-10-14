// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMLibDefines.h"
#include "Core/ITMBasicEngine.tpp"
#include "Core/ITMMultiEngine.tpp"
#include "Core/ITMDenseMapper.tpp"
#include "Engines/Meshing/CPU/ITMMeshingEngine_CPU.tpp"
#include "Engines/Meshing/CPU/ITMMultiMeshingEngine_CPU.tpp"
#include "Engines/MultiScene/ITMMapGraphManager.tpp"
#include "Engines/Visualisation/CPU/ITMMultiVisualisationEngine_CPU.tpp"
#include "Engines/Reconstruction/CPU/ITMSceneReconstructionEngine_CPU.tpp"
#include "Engines/Swapping/CPU/ITMSwappingEngine_CPU.tpp"
#include "Engines/Visualisation/CPU/ITMVisualisationEngine_CPU.tpp"
#include "Engines/Visualisation/Interface/ITMVisualisationEngine.h"
#include "Trackers/ITMTrackerFactory.h"

namespace ITMLib
{
	template class ITMVoxelMapGraphManager<ITMVoxel>;
	template class ITMSwappingEngine_CPU<ITMVoxel>;
}
