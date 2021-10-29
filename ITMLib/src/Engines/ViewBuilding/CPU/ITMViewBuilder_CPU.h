// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Engines/ITMViewBuilder.h>

namespace ITMLib
{
	class ITMViewBuilder_CPU : public ITMViewBuilder
	{
	public:
		void ConvertDisparityToDepth(ITMFloatImage *depth_out, const ITMShortImage *disp_in, const ITMIntrinsics *depthIntrinsics,
			Vector2f disparityCalibParams) override;
		void ConvertDepthAffineToFloat(ITMFloatImage *depth_out, const ITMShortImage *depth_in, Vector2f depthCalibParams) override;

		void DepthFiltering(ITMFloatImage *image_out, const ITMFloatImage *image_in) override;
		void NormalFiltering(ITMFloat4Image *normals_out, const ITMFloat4Image *normals_in) override;
		void ComputeNormalAndWeights(ITMFloat4Image *normal_out, const ITMFloatImage *depth_in, Vector4f intrinsic) override;

		void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, bool useBilateralFilter, bool computeNormals = false) override;
		void UpdateView(ITMView **view, ITMUChar4Image *rgbImage, ITMShortImage *depthImage, bool useBilateralFilter, ITMIMUMeasurement *imuMeasurement, bool computeNormals = false) override;

		explicit ITMViewBuilder_CPU(const ITMRGBDCalib& calib);
		~ITMViewBuilder_CPU() override;
	};
}

