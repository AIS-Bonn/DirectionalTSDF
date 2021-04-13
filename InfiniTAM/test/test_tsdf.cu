#include <test/catch2/catch.hpp>

#include <ITMLib/Objects/Scene/TSDF.h>
#include <ITMLib/Objects/Scene/TSDF_CPU.h>
#include <ITMLib/Objects/Scene/TSDF_CUDA.h>
#include <ITMLib/ITMLibDefines.h>

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

TEST_CASE("GPU readVoxel", "TSDF")
{
	printf("===== GPU =======\n");
	TSDF_CUDA<ITMVoxel, IndexShort> tsdf(1000);

	readVoxel<<<1, 1>>>(tsdf.getMap());
	ORcudaKernelCheck
}

TEST_CASE("CPU readVoxel", "TSDF")
{
	printf("===== CPU =======\n");
	TSDF_CPU<ITMVoxel, IndexShort> tsdf(1000);

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
