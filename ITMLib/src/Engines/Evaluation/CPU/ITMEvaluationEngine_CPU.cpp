//
// Created by Malte Splietker on 28.12.22.
//

#include "ITMEvaluationEngine_CPU.h"

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <ITMLib/Utils/ITMMath.h>
#include "../Shared/ITMEvaluationEngine_Shared.h"
#include <Trackers/Shared/ITMICPTracker_Shared.h>

using namespace ITMLib;

ITMEvaluationEngine_CPU::ITMEvaluationEngine_CPU()
{}

ITMRenderError ITMEvaluationEngine_CPU::ComputeICPError(
	const ITMFloatImage& depth,
	const ORUtils::Image<ORUtils::Vector4<float>>& points,
	const ORUtils::Image<ORUtils::Vector4<float>>& normals,
	const Matrix4f& depthImageInvPose,
	const Matrix4f& sceneRenderingPose,
	ITMIntrinsics& intrinsics_d
) const
{
	Vector2i imgSize = intrinsics_d.imgSize;

	const float* depthPtr = depth.GetData(MEMORYDEVICE_CPU);
	const Vector4f* pointsRay = points.GetData(MEMORYDEVICE_CPU);
	const Vector4f* normalsRay = normals.GetData(MEMORYDEVICE_CPU);

	std::vector<float> errors, icpErrors;
	for (int x = 0; x < imgSize.width; x++)
		for (int y = 0; y < imgSize.height; y++)
		{
			float A[6];
			float error, icpError;
			float weight;
			bool isValidPoint = computePerPointGH_Depth_Ab<false, false>(
				A, icpError, weight, x, y,
				depthImageInvPose, sceneRenderingPose,
				depthPtr,
				intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
				intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
				pointsRay, normalsRay, 100.0);

			isValidPoint &= computePerPointError<false, false>(
				error, x, y, depthPtr,
				intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
				intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
				depthImageInvPose, sceneRenderingPose,
				pointsRay);

			if (!isValidPoint)
				continue;

			errors.push_back(std::fabs(error));
			icpErrors.push_back(std::fabs(icpError));
		}

	ITMRenderError result;
	result.MAE = thrust::reduce(errors.begin(), errors.end(), (float) 0, thrust::plus<float>()) / errors.size();
	result.RMSE = std::sqrt(thrust::transform_reduce(errors.begin(), errors.end(),
	                                                 square<float>(),
	                                                 (float) 0,
	                                                 thrust::plus<float>()) / errors.size());
	result.icpMAE =
		thrust::reduce(icpErrors.begin(), icpErrors.end(), (float) 0, thrust::plus<float>()) / icpErrors.size();
	result.icpRMSE = std::sqrt(thrust::transform_reduce(icpErrors.begin(), icpErrors.end(),
	                                                    square<float>(),
	                                                    (float) 0,
	                                                    thrust::plus<float>()) / icpErrors.size());
	return result;
}

ITMRenderError
ITMEvaluationEngine_CPU::ComputePhotometricError(const ITMUChar4Image& image, const ITMUChar4Image& render,
                                                 const ITMFloat4Image& points, const Matrix4f& depthImageInvPose,
                                                 const Matrix4f& sceneRenderingPose,
                                                 ITMIntrinsics& intrinsics_rgb) const
{
	const Vector4u* imagePtr = image.GetData(MEMORYDEVICE_CPU);
	const Vector4u* renderPtr = render.GetData(MEMORYDEVICE_CPU);
	const Vector4f* pointsRay = points.GetData(MEMORYDEVICE_CPU);

	std::vector<float> errors;
	for (int x = 0; x < intrinsics_rgb.imgSize.width; x++)
		for (int y = 0; y < intrinsics_rgb.imgSize.height; y++)
		{
			float error;

			int idx = PixelCoordsToIndex(x, y, intrinsics_rgb.imgSize);

			float intensityImage =
				(0.299f * imagePtr[idx].x + 0.587f * imagePtr[idx].y + 0.114f * imagePtr[idx].z) / 255.f;
			float intensityRender =
				(0.299f * renderPtr[idx].x + 0.587f * renderPtr[idx].y + 0.114f * renderPtr[idx].z) / 255.f;

			error = std::fabs(intensityImage - intensityRender);

			bool isValidPoint = intensityRender > 0 and pointsRay[idx].w >= 0;
			if (!isValidPoint)
			{
				continue;
			}

			errors.push_back(std::fabs(error));
		}

	ITMRenderError result;
	result.MAE = thrust::reduce(errors.begin(), errors.end(), (float) 0, thrust::plus<float>()) / errors.size();
	result.RMSE = std::sqrt(thrust::transform_reduce(errors.begin(), errors.end(),
	                                                 square<float>(),
	                                                 (float) 0,
	                                                 thrust::plus<float>()) / errors.size());
	result.icpMAE = 0;
	result.icpRMSE = 0;

	return result;
}
