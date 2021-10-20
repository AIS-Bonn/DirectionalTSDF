// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMViewBuilder.h"

namespace ITMLib
{
	class ITMViewBuilder_CUDA : public ITMViewBuilder
	{
	public:
		void ConvertDisparityToDepth(ITMFloatImage *depth_out, const ITMShortImage *depth_in, const ITMIntrinsics *depthIntrinsics, 
			Vector2f disparityCalibParams);
		void ConvertDepthAffineToFloat(ITMFloatImage *depth_out, const ITMShortImage *depth_in, Vector2f depthCalibParams);

		void DepthFiltering(ITMFloatImage *image_out, const ITMFloatImage *image_in);
		void NormalFiltering(ITMFloat4Image *normals_out, const ITMFloat4Image *normals_in);
		void ComputeNormalAndWeights(ITMFloat4Image *normal_out, const ITMFloatImage *depth_in, Vector4f intrinsic);

		void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, bool useBilateralFilter, bool computeNormals = false);
		void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *depthImage, bool useBilateralFilter, ITMIMUMeasurement *imuMeasurement, bool computeNormals = false);

		ITMViewBuilder_CUDA(const ITMRGBDCalib& calib);
		~ITMViewBuilder_CUDA(void);
	};
}
