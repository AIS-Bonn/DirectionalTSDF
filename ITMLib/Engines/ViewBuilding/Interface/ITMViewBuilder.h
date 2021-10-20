// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLib/Objects/Camera/ITMRGBDCalib.h"
#include "ITMLib/Objects/Views/ITMViewIMU.h"
#include "ITMLib/Objects/Stats/ITMPreprocessingTimeStats.h"

namespace ITMLib
{
	/** \brief
	*/
	class ITMViewBuilder
	{
	protected:
		const ITMRGBDCalib calib;
		ITMShortImage *shortImage;
		ITMFloatImage *floatImage;
		ITMFloat4Image *normals;

		ITMPreprocessingTimeStats timeStats;

	public:
		virtual void ConvertDisparityToDepth(ITMFloatImage *depth_out, const ITMShortImage *disp_in, const ITMIntrinsics *depthIntrinsics,
			Vector2f disparityCalibParams) = 0;
		virtual void ConvertDepthAffineToFloat(ITMFloatImage *depth_out, const ITMShortImage *depth_in, Vector2f depthCalibParams) = 0;

		virtual void DepthFiltering(ITMFloatImage *image_out, const ITMFloatImage *image_in) = 0;
		virtual void NormalFiltering(ITMFloat4Image *normals_out, const ITMFloat4Image *normals_in) = 0;
		virtual void ComputeNormalAndWeights(ITMFloat4Image *normal_out, const ITMFloatImage *depth_in, Vector4f intrinsic) = 0;

		virtual void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, bool useBilateralFilter, bool computeNormals = false) = 0;
		virtual void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *depthImage, bool useBilateralFilter, ITMIMUMeasurement *imuMeasurement, bool computeNormals = false) = 0;

		const ITMPreprocessingTimeStats &GetTimeStats() const
		{
			return timeStats;
		}

		ITMViewBuilder(const ITMRGBDCalib& calib_)
		: calib(calib_)
		{
			this->shortImage = nullptr;
			this->floatImage = nullptr;
			this->normals = nullptr;
		}

		virtual ~ITMViewBuilder()
		{
			if (this->shortImage != nullptr) delete this->shortImage;
			if (this->floatImage != nullptr) delete this->floatImage;
			if (this->normals != nullptr) delete this->normals;
		}
	};
}
