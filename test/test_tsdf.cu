#include <test/catch2/catch.hpp>

#include <ITMLib/Objects/Scene/TSDF.h>
#include <ITMLib/Objects/Scene/TSDF_CPU.h>
#include <ITMLib/Objects/Scene/TSDF_CUDA.h>
#include <ITMLib/ITMLibDefines.h>

#include <unordered_set>
#include <stdgpu/unordered_set.cuh>

using namespace ITMLib;

template<typename TVoxel, typename TIndex>
__global__
void readVoxel(stdgpu::unordered_map<TIndex, TVoxel*> tsdf)
{
	bool found;
	float confidence;
	readVoxel(found, tsdf, Vector3i(0, 0, 0));
	readFromSDF_float_uninterpolated(found, tsdf, Vector3f(0.5, 9, 1.5));
	readFromSDF_float_interpolated(found, tsdf, Vector3f(0.5, 9, 1.5));
	readFromSDF_float_interpolated(found, tsdf, Vector3f(0.5, 9, 1.5), 1);
	readWithConfidenceFromSDF_float_uninterpolated(found, confidence, tsdf, Vector3f(0.5, 9, 1.5), 1);
	readWithConfidenceFromSDF_float_interpolated(found, confidence, tsdf, Vector3f(0.5, 9, 1.5), 1);
}

TEST_CASE("GPU readVoxel", "TSDFBase")
{
	printf("===== GPU =======\n");
	TSDF_CUDA<IndexShort, ITMVoxel> tsdf(1000);

	readVoxel<<<1, 1>>>(tsdf.getMap());
	ORcudaKernelCheck
}


template<template<typename...> class Set, typename... Args>
_CPU_AND_GPU_CODE_
void inline function(Set<ITMIndex, Args...> set)
{
//	set.insert(ITMIndex(threadIdx.x));
}

__global__
void setTest(stdgpu::unordered_set<ITMIndex> s)
{
	function(s);
}

TEST_CASE("stdgpu", "TSDFBase")
{
	stdgpu::unordered_set<ITMIndex> s = stdgpu::unordered_set<ITMIndex>::createDeviceObject(10000);
	setTest<<<50, 10>>>(s);
	REQUIRE(s.size() == 10);

//	std::unordered_set<int> s2;
//	function(s2);
}

TEST_CASE("CPU readVoxel", "TSDFBase")
{
	printf("===== CPU =======\n");
	TSDF_CPU<IndexShort, ITMVoxel> tsdf(1000);

	bool found;
	float sdf, confidence;
	readVoxel(found, tsdf.getMap(), Vector3i(0, 0, 0));
	sdf = readFromSDF_float_uninterpolated(found, tsdf.getMap(), Vector3f(0.5, 9, 1.5));
	sdf = readFromSDF_float_interpolated(found, tsdf.getMap(), Vector3f(0.5, 9, 1.5));
	sdf = readFromSDF_float_interpolated(found, tsdf.getMap(), Vector3f(0.5, 9, 1.5), 1);
	sdf = readWithConfidenceFromSDF_float_uninterpolated(found, confidence, tsdf.getMap(), Vector3f(0.5, 9, 1.5), 1);
	sdf = readWithConfidenceFromSDF_float_interpolated(found, confidence, tsdf.getMap(), Vector3f(0.5, 9, 1.5), 1);
//	REQUIRE(sdf == 1.0f);
}
