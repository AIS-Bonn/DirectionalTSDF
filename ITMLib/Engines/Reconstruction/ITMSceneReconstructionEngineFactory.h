// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "CPU/ITMSceneReconstructionEngine_CPU.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "CUDA/ITMSceneReconstructionEngine_CUDA.h"
#endif

namespace ITMLib
{

/**
 * \brief This struct provides functions that can be used to construct scene reconstruction engines.
 */
struct ITMSceneReconstructionEngineFactory
{
  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

  /**
   * \brief Makes a scene reconstruction engine.
   *
   * \param deviceType  The device on which the scene reconstruction engine should operate.
   */
  static ITMSceneReconstructionEngine *MakeSceneReconstructionEngine(
    const std::shared_ptr<const ITMLibSettings>& settings)
  {
    ITMSceneReconstructionEngine *sceneRecoEngine = nullptr;

    switch(settings->deviceType)
    {
      case ITMLibSettings::DEVICE_CPU:
        sceneRecoEngine = new ITMSceneReconstructionEngine_CPU(settings);
        break;
      case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
        sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA(settings);
#endif
        break;
    }

    return sceneRecoEngine;
  }
};

}
