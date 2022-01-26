// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMViewBuilder_CUDA.h"

#include <Engines/ViewBuilding/Shared/ITMViewBuilder_Shared.h>
#include <Utils/ITMTimer.h>
#include <ORUtils/CUDADefines.h>
#include <ORUtils/MemoryBlock.h>
#include <ITMLib/Core/ITMConstants.h>

using namespace ITMLib;
using namespace ORUtils;

ITMViewBuilder_CUDA::ITMViewBuilder_CUDA(const ITMRGBDCalib& calib) : ITMViewBuilder(calib)
{}

ITMViewBuilder_CUDA::~ITMViewBuilder_CUDA(void)
{}

//---------------------------------------------------------------------------
//
// kernel function declaration 
//
//---------------------------------------------------------------------------


__global__ void
convertDisparityToDepth_device(float* depth_out, const short* depth_in, Vector2f disparityCalibParams, float fx_depth,
                               Vector2i imgSize, bool filterDepth);

__global__ void
convertDepthAffineToFloat_device(float* d_out, const short* d_in, Vector2i imgSize, Vector2f depthCalibParams,
                                 bool filterDepth);

__global__ void filterDepthBilateral_device(float* imageData_out, const float* imageData_in, Vector2i imgDims);

__global__ void filterNormalsBilateral_device(Vector4f* normals_out, const Vector4f* normals_in, Vector2i imgDims);

__global__ void
ComputeNormalAndWeight_device(const float* depth_in, Vector4f* normal_out, Vector2i imgDims, Vector4f intrinsic);

//---------------------------------------------------------------------------
//
// host methods
//
//---------------------------------------------------------------------------

void ITMViewBuilder_CUDA::UpdateView(ITMView** view_ptr, ITMUChar4Image* rgbImage, ITMShortImage* rawDepthImage,
                                     bool useDepthFilter, bool useBilateralFilter, bool computeNormals)
{
	static int count = 0;
	timeStats.Reset();
	ITMTimer timer;
	timer.Tick();

	if (*view_ptr == nullptr)
	{
		*view_ptr = new ITMView(calib, rgbImage->noDims, rawDepthImage->noDims, true);
		delete this->shortImage;
		this->shortImage = new ITMShortImage(rawDepthImage->noDims, true, true);
		delete this->floatImage;
		this->floatImage = new ITMFloatImage(rawDepthImage->noDims, true, true);
		delete this->normals;
		this->normals = new ITMFloat4Image(rawDepthImage->noDims, true, true);
	}

	ITMView* view = *view_ptr;

	view->rgb->SetFrom(rgbImage, ORUtils::CPU_TO_CUDA);
	this->shortImage->SetFrom(rawDepthImage, ORUtils::CPU_TO_CUDA);

	Vector2f affine = view->calib.disparityCalib.GetParams();

#if SCALE_EXPERIMENT == 2
//	float scales[] = {1, 1.1, 0.95, 1.05, 0.9};
	float scales[] = {1, 1.05, 0.975, 1.025, 0.95};

	affine.x /= scales[count % SCALE_EXPERIMENT_NUM_SENSORS];

	// change scaling every Nth frame
	count++;
	if (count > 0 and count % 20 == 5)
	{
		// change scaling
//		affine.x /= 1.1;

		// cut off part of image
//		this->shortImage->UpdateHostFromDevice();
//		short* data = this->shortImage->GetData(MEMORYDEVICE_CPU);
//		for (int x = 0; x < 300; x++)
//		{
//			for (int y = 0; y < this->shortImage->noDims.height; y++)
//			{
//				int idx = PixelCoordsToIndex(x, y, this->shortImage->noDims);
//				data[idx] = 0;
//			}
//		}
//		this->shortImage->UpdateDeviceFromHost();
	}
#endif


	switch (view->calib.disparityCalib.GetType())
	{
		case ITMDisparityCalib::TRAFO_KINECT:
			this->ConvertDisparityToDepth(view->depth, this->shortImage, &(view->calib.intrinsics_d),
			                              view->calib.disparityCalib.GetParams(), useDepthFilter);
			break;
		case ITMDisparityCalib::TRAFO_AFFINE:
			this->ConvertDepthAffineToFloat(view->depth, this->shortImage, affine, useDepthFilter);
			break;
		default:
			break;
	}
	timeStats.copyImages += timer.Tock();

	timer.Tick();
	this->floatImage->ChangeDims(view->depth->noDims);
	this->normals->ChangeDims(view->depth->noDims);
	this->DepthBilateralFiltering(this->floatImage, view->depth);
	if (useBilateralFilter)
	{ // user filtered depth image
		view->depth->SetFrom(this->floatImage, ORUtils::CUDA_TO_CUDA);
	}
	timeStats.bilateralFilter = timer.Tock();

	if (computeNormals)
	{
		timer.Tick();
#define FILTER_NORMALS
#ifdef FILTER_NORMALS
		this->ComputeNormalAndWeights(this->normals, this->floatImage, // use pre-filtered depth image
		                              view->calib.intrinsics_d.projectionParamsSimple.all);
		this->NormalBilateralFiltering(view->depthNormal, this->normals);
#else
		// normals from filteres image
		this->ComputeNormalAndWeights(view->depthNormal, view->depth,
																	view->calib.intrinsics_d.projectionParamsSimple.all);
#endif
		timeStats.normalEstimation = timer.Tock();
	}
}

void ITMViewBuilder_CUDA::UpdateView(ITMView** view_ptr, ITMUChar4Image* rgbImage, ITMShortImage* depthImage,
                                     bool useDepthFilter, bool useBilateralFilter, ITMIMUMeasurement* imuMeasurement,
                                     bool computeNormals)
{
	if (*view_ptr == nullptr)
	{
		*view_ptr = new ITMViewIMU(calib, rgbImage->noDims, depthImage->noDims, true);
		delete this->shortImage;
		this->shortImage = new ITMShortImage(depthImage->noDims, true, true);
		delete this->floatImage;
		this->floatImage = new ITMFloatImage(depthImage->noDims, true, true);
		delete this->normals;
		this->normals = new ITMFloat4Image(depthImage->noDims, true, true);
	}

	ITMViewIMU* imuView = (ITMViewIMU*) (*view_ptr);
	imuView->imu->SetFrom(imuMeasurement);

	this->UpdateView(view_ptr, rgbImage, depthImage, useDepthFilter, useBilateralFilter, computeNormals);
}

void ITMViewBuilder_CUDA::ConvertDisparityToDepth(ITMFloatImage* depth_out, const ITMShortImage* depth_in,
                                                  const ITMIntrinsics* depthIntrinsics,
                                                  Vector2f disparityCalibParams, bool filterDepth)
{
	Vector2i imgSize = depth_in->noDims;

	const short* d_in = depth_in->GetData(MEMORYDEVICE_CUDA);
	float* d_out = depth_out->GetData(MEMORYDEVICE_CUDA);

	float fx_depth = depthIntrinsics->projectionParamsSimple.fx;

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) imgSize.x / (float) blockSize.x),
	              (int) ceil((float) imgSize.y / (float) blockSize.y));

	convertDisparityToDepth_device << <
	gridSize, blockSize >> >(d_out, d_in, disparityCalibParams, fx_depth, imgSize, filterDepth);
	ORcudaKernelCheck;
}

void ITMViewBuilder_CUDA::ConvertDepthAffineToFloat(ITMFloatImage* depth_out, const ITMShortImage* depth_in,
                                                    Vector2f depthCalibParams, bool filterDepth)
{
	Vector2i imgSize = depth_in->noDims;

	const short* d_in = depth_in->GetData(MEMORYDEVICE_CUDA);
	float* d_out = depth_out->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) imgSize.x / (float) blockSize.x),
	              (int) ceil((float) imgSize.y / (float) blockSize.y));

	convertDepthAffineToFloat_device << < gridSize, blockSize >> >(d_out, d_in, imgSize, depthCalibParams, filterDepth);
	ORcudaKernelCheck;
}

void ITMViewBuilder_CUDA::DepthBilateralFiltering(ITMFloatImage* image_out, const ITMFloatImage* image_in)
{
	Vector2i imgDims = image_in->noDims;

	const float* imageData_in = image_in->GetData(MEMORYDEVICE_CUDA);
	float* imageData_out = image_out->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) imgDims.x / (float) blockSize.x),
	              (int) ceil((float) imgDims.y / (float) blockSize.y));

	filterDepthBilateral_device << < gridSize, blockSize >> >(imageData_out, imageData_in, imgDims);
	ORcudaKernelCheck;
}

void ITMViewBuilder_CUDA::ComputeNormalAndWeights(ITMFloat4Image* normal_out, const ITMFloatImage* depth_in,
                                                  Vector4f intrinsic)
{
	Vector2i imgDims = depth_in->noDims;

	const float* depthData_in = depth_in->GetData(MEMORYDEVICE_CUDA);

	Vector4f* normalData_out = normal_out->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) imgDims.x / (float) blockSize.x),
	              (int) ceil((float) imgDims.y / (float) blockSize.y));

	ComputeNormalAndWeight_device << < gridSize, blockSize >> >(depthData_in, normalData_out, imgDims, intrinsic);
	ORcudaKernelCheck;
}

void ITMViewBuilder_CUDA::NormalBilateralFiltering(ITMFloat4Image* normals_out, const ITMFloat4Image* normals_in)
{
	Vector2i imgDims = normals_in->noDims;

	const Vector4f* n_in = normals_in->GetData(MEMORYDEVICE_CUDA);
	Vector4f* n_out = normals_out->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) imgDims.x / (float) blockSize.x),
	              (int) ceil((float) imgDims.y / (float) blockSize.y));

	filterNormalsBilateral_device << < gridSize, blockSize >> >(n_out, n_in, imgDims);
	ORcudaKernelCheck;
}

//---------------------------------------------------------------------------
//
// kernel function implementation
//
//---------------------------------------------------------------------------

__global__ void
convertDisparityToDepth_device(float* d_out, const short* d_in, Vector2f disparityCalibParams, float fx_depth,
                               Vector2i imgSize, bool filterDepth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= imgSize.x) || (y >= imgSize.y)) return;

	convertDisparityToDepth(d_out, x, y, d_in, disparityCalibParams, fx_depth, imgSize, filterDepth);
}

__global__ void
convertDepthAffineToFloat_device(float* d_out, const short* d_in, Vector2i imgSize, Vector2f depthCalibParams,
                                 bool filterDepth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < 0 || y < 0 || x >= imgSize.x || y >= imgSize.y) return;

	convertDepthAffineToFloat(d_out, x, y, d_in, imgSize, depthCalibParams, filterDepth);
}

__global__ void filterDepthBilateral_device(float* imageData_out, const float* imageData_in, Vector2i imgDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= imgDims.x || y >= imgDims.y) return;

	filterDepthBilateral(imageData_out, imageData_in, 5.0, 0.025, x, y, imgDims);
}

__global__ void filterNormalsBilateral_device(Vector4f* normals_out, const Vector4f* normals_in, Vector2i imgDims)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= imgDims.x || y >= imgDims.y) return;

	filterNormalsBilateral(normals_out, normals_in, 2.5, 5.0, x, y, imgDims);
}

__global__ void
ComputeNormalAndWeight_device(const float* depth_in, Vector4f* normal_out, Vector2i imgDims, Vector4f intrinsic)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < 1 || y < 1 || x >= imgDims.x - 1 || y >= imgDims.y - 1)
		return;

	computeNormalAndWeight(depth_in, normal_out, x, y, imgDims, intrinsic);
}