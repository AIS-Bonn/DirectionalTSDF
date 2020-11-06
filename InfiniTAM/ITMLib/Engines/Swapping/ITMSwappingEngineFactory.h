// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "CPU/ITMSwappingEngine_CPU.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "CUDA/ITMSwappingEngine_CUDA.h"
#endif

namespace ITMLib
{

/**
 * \brief This struct provides functions that can be used to construct swapping engines.
 */
struct ITMSwappingEngineFactory
{
  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

  /**
   * \brief Makes a swapping engine.
   *
   * \param deviceType  The device on which the swapping engine should operate.
   */
  template <typename TVoxel>
  static ITMSwappingEngine<TVoxel> *MakeSwappingEngine(ITMLibSettings::DeviceType deviceType)
  {
    ITMSwappingEngine<TVoxel> *swappingEngine = nullptr;

    switch(deviceType)
    {
      case ITMLibSettings::DEVICE_CPU:
        swappingEngine = new ITMSwappingEngine_CPU<TVoxel>;
        break;
      case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
        swappingEngine = new ITMSwappingEngine_CUDA<TVoxel>;
#endif
        break;
    }

    return swappingEngine;
  }
};

}
