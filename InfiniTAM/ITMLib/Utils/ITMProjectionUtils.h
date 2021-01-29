// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once
#include <cmath>

#include "ITMMath.h"
#include "ORUtils/PlatformIndependence.h"

namespace ITMLib
{

template<typename T> _CPU_AND_GPU_CODE_ inline Vector2f project(const THREADPTR(ORUtils::Vector3<T>) &point_3d, const THREADPTR(Vector4f) &intrinsics)
{
	return Vector2f(intrinsics.x * (float)point_3d.x / (float)point_3d.z + intrinsics.z,
					intrinsics.y * (float)point_3d.y / (float)point_3d.z + intrinsics.w);
}

_CPU_AND_GPU_CODE_
inline Vector3f reprojectImagePoint(float x, float y, float depth, const Vector4f &invProjParams)
{
	Vector3f pt_camera;
	pt_camera.z = depth;
	pt_camera.x = pt_camera.z * ((x - invProjParams.z) * invProjParams.x);
	pt_camera.y = pt_camera.z * ((y - invProjParams.w) * invProjParams.y);

	return pt_camera;
}

template<typename T> _CPU_AND_GPU_CODE_ inline Vector3f unproject(const THREADPTR(T) x, const THREADPTR(T) y, const THREADPTR(float) depth, const THREADPTR(Vector4f) &intrinsics)
{
	return Vector3f(depth * (((float)x - intrinsics.z) / intrinsics.x),
					depth * (((float)y - intrinsics.w) / intrinsics.y),
					depth);
}

_CPU_AND_GPU_CODE_
inline Vector4f invertProjectionParams(Vector4f projParams)
{
	return Vector4f(1.0f / projParams.x, 1.0f / projParams.y, projParams.z, projParams.w);
}

} // namespace ITMLib
