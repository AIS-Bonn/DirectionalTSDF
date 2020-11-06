// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "CPU/ITMVisualisationEngine_CPU.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "CUDA/ITMVisualisationEngine_CUDA.h"
#endif

namespace ITMLib
{

/**
 * \brief This struct provides functions that can be used to construct visualisation engines.
 */
struct ITMVisualisationEngineFactory
{
  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

  /**
   * \brief Makes a visualisation engine.
   *
   * \param deviceType  The device on which the visualisation engine should operate.
   */
  static ITMVisualisationEngine *MakeVisualisationEngine(
  	ITMLibSettings::DeviceType deviceType, const std::shared_ptr<const ITMLibSettings>& settings)
  {
    ITMVisualisationEngine *visualisationEngine = nullptr;

    switch(deviceType)
    {
      case ITMLibSettings::DEVICE_CPU:
        visualisationEngine = new ITMVisualisationEngine_CPU(settings);
        break;
      case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
        visualisationEngine = new ITMVisualisationEngine_CUDA(settings);
#endif
        break;
    }

    return visualisationEngine;
  }
};

}
