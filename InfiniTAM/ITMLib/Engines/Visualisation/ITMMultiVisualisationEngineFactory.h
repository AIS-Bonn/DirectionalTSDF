// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "CPU/ITMMultiVisualisationEngine_CPU.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "CUDA/ITMMultiVisualisationEngine_CUDA.h"
#endif

namespace ITMLib
{

	/**
	 * \brief This struct provides functions that can be used to construct visualisation engines.
	 */
	struct ITMMultiVisualisationEngineFactory
	{
		//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

		/**
		 * \brief Makes a visualisation engine.
		 *
		 * \param deviceType  The device on which the visualisation engine should operate.
		 */
		static ITMMultiVisualisationEngine *MakeVisualisationEngine(ITMLibSettings::DeviceType deviceType, const std::shared_ptr<const ITMLibSettings>& settings)
		{
			ITMMultiVisualisationEngine *visualisationEngine = nullptr;

			switch (deviceType)
			{
			case ITMLibSettings::DEVICE_CPU:
				visualisationEngine = new ITMMultiVisualisationEngine_CPU(settings);
				break;
			case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
				visualisationEngine = new ITMMultiVisualisationEngine_CUDA(settings);
#endif
				break;
			}

			return visualisationEngine;
		}
	};

}
