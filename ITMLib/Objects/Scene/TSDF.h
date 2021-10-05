//
// Created by Malte Splietker on 08.04.21.
//

#pragma once

#include <ITMLib/Utils/ITMMath.h>
#include "ITMDirectional.h"
#include "ITMMultiSceneAccess.h"

namespace ITMLib
{

template<class T>
class Index : public ORUtils::Vector3<T>
{
public:
	_CPU_AND_GPU_CODE_ Index() : ORUtils::Vector3<T>()
	{}

	_CPU_AND_GPU_CODE_ Index(const ORUtils::Vector3<T>& u)
		: ORUtils::Vector3<T>(u)
	{}

	_CPU_AND_GPU_CODE_ inline ORUtils::Vector3<int> getPosition() const
	{
		return this->toInt();
	}

	_CPU_AND_GPU_CODE_ inline TSDFDirection getDirection() const
	{
		return TSDFDirection::NONE;
	}
};

template<class T>
class IndexDirectional : public ORUtils::Vector4<T>
{
public:
	_CPU_AND_GPU_CODE_ IndexDirectional() : ORUtils::Vector4<T>()
	{}

	_CPU_AND_GPU_CODE_ IndexDirectional(const ORUtils::Vector3_<T>& u, TSDFDirection direction)
		: ORUtils::Vector4<T>(u, TSDFDirection_type(direction))
	{}

	_CPU_AND_GPU_CODE_ inline ORUtils::Vector3<int> getPosition() const
	{
		return this->toVector3().toInt();
	}

	_CPU_AND_GPU_CODE_ inline TSDFDirection getDirection() const
	{
		return TSDFDirection(this->w);
	}
};

typedef class IndexDirectional<short> IndexDirectionalShort;
typedef class Index<short> IndexShort;

template <typename T>
_CPU_AND_GPU_CODE_ inline void voxelIdxToIndexAndOffset(
	Index<T>& index, unsigned short& offset, const Vector3i& voxelIdx,
	const TSDFDirection direction = TSDFDirection::NONE)
{
	Vector3i blockIdx;
	voxelToBlockPosAndOffset(voxelIdx, blockIdx, offset);
	index = Index<T>(blockIdx.toShort());
}

template <typename T>
_CPU_AND_GPU_CODE_ inline void voxelIdxToIndexAndOffset(
	IndexDirectional<T>& index, unsigned short& offset, const Vector3i& voxelIdx,
	const TSDFDirection direction = TSDFDirection::NONE)
{
	Vector3i blockIdx;
	voxelToBlockPosAndOffset(voxelIdx, blockIdx, offset);
	index = IndexDirectional<T>(blockIdx.toShort(), direction);
}

struct AllocationStats
{
	unsigned int noAllocationsPerDirection[N_DIRECTIONS];
	unsigned long long noAllocations;

	AllocationStats()
	{
		noAllocations = 0;
		memset(noAllocationsPerDirection, 0, sizeof(noAllocationsPerDirection));
	}
};

template<typename TIndex, typename TVoxel>
class TSDF_CUDA;
template<typename TIndex, typename TVoxel>
class TSDF_CPU;

template<typename TIndex, typename TVoxel>
class TSDF
{
public:
	virtual void resize(size_t newSize) = 0;

	virtual void clear() = 0;

	virtual size_t size() = 0;

	TSDF_CUDA<TIndex, TVoxel>* toCUDA()
	{
		return dynamic_cast<TSDF_CUDA<TIndex, TVoxel>*>(this);
	}

	TSDF_CPU<TIndex, TVoxel>* toCPU()
	{
		return dynamic_cast<TSDF_CPU<TIndex, TVoxel>*>(this);
	}

	virtual  MemoryDeviceType deviceType() = 0;

	size_t allocatedBlocksMax = 0;
	TVoxel* voxels = nullptr;

	AllocationStats allocationStats;
};

/**
 * Abstract interface for TSDFBase.
 * in Init() the map is to be initialized (and entries reset) with the given visible entries
 * @tparam TVoxel voxel type
 * @tparam Map map type. Either std::unordered_map or stdgpu::unordered_map
 */
template<typename TIndex, typename TVoxel, template<typename, typename...> class Map>
class TSDFBase : public TSDF<TIndex, TVoxel>
{
public:
	virtual ~TSDFBase() = default;

	virtual void allocate(const TIndex* blocks, size_t N) = 0;

	void clear() override
	{
		this->allocationStats.noAllocations = 0;
		map.clear();
	}

	size_t size() override
	{
		return map.size();
	}

	inline Map<TIndex, TVoxel*>& getMap()
	{ return map; }

protected:
	Map<TIndex, TVoxel*> map;
};

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_uninterpolated(
	bool& found, const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point,
	const TSDFDirection direction = TSDFDirection::NONE)
{
	TVoxel res = readVoxel(found, tsdf, Vector3i((int) ROUND(point.x), (int) ROUND(point.y), (int) ROUND(point.z)),
	                       direction);
	return res.sdf;
}

#define DISCARD_ZERO_WEIGHT if (voxel.w_depth <= 0) return 1;

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_interpolated(
	bool& found, const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point,
	const TSDFDirection direction = TSDFDirection::NONE)
{
	float res1, res2, v1, v2;
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);
	TVoxel voxel;

	found = false;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0), direction);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0), direction);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0), direction);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0), direction);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1), direction);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1), direction);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1), direction);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1), direction);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	found = true;
	return TVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_interpolated(
	bool& found, const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point,
	int maxW, const TSDFDirection direction = TSDFDirection::NONE)
{
	float res1, res2, v1, v2;
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);

	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0), direction);
		v1 = v.sdf;
		maxW = v.w_depth;
	}
	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0), direction);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0), direction);
		v1 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0), direction);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1), direction);
		v1 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1), direction);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1), direction);
		v1 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const TVoxel& v = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1), direction);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	found = true;
	return TVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline float readWithConfidenceFromSDF_float_uninterpolated(
	bool& found, float& confidence, const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point,
	const int maxW, const TSDFDirection direction = TSDFDirection::NONE)
{
	TVoxel res = readVoxel(found, tsdf,
	                       Vector3i((int) ROUND(point.x), (int) ROUND(point.y), (int) ROUND(point.z)), direction);
	if (found)
		confidence = TVoxel::weightToFloat(res.w_depth, maxW);
	else
		confidence = 0;

	return TVoxel::valueToFloat(res.sdf);
}

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline float readWithConfidenceFromSDF_float_interpolated(
	bool& found, float& confidence, const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point,
	const int maxW, const TSDFDirection direction = TSDFDirection::NONE)
{
	float res1, res2, v1, v2;
	float res1_c, res2_c, v1_c, v2_c;
	TVoxel voxel;

	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);

	found = false;
	confidence = 0;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0), direction);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0), direction);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res1_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0), direction);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0), direction);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res1_c = (1.0f - coeff.y) * res1_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1), direction);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1), direction);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res2_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1), direction);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1), direction);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res2_c = (1.0f - coeff.y) * res2_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	found = true;
	confidence = TVoxel::weightToFloat((1.0f - coeff.z) * res1_c + coeff.z * res2_c, maxW);

	return TVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_uninterpolated(
	float& confidence, const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point,
	const int maxW, const TSDFDirection direction = TSDFDirection::NONE)
{
	Vector3f color(0.0f);
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);

	confidence = 0;

	bool found;
	TVoxel voxel = readVoxel(found, tsdf, pos, direction);
	confidence = TVoxel::weightToFloat(voxel.w_color, maxW);

	return Vector4f(voxel.clr.toFloat() / 255.0f, 255.0f);
}

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(
	float& confidence, const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point,
	const int maxW, const TSDFDirection direction = TSDFDirection::NONE)
{
	TVoxel voxel;
	Vector3f color(0.0f);
	float w_color = 0;
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);

	bool found;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0), direction);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0), direction);
	color += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0), direction);
	color += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0), direction);
	color += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1), direction);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.w_color;

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1), direction);
	color += (coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.w_color;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1), direction);
	color += (1.0f - coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (coeff.y) * coeff.z * voxel.w_color;

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1), direction);
	color += (coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (coeff.x) * (coeff.y) * coeff.z * voxel.w_color;

	confidence = TVoxel::weightToFloat(w_color, maxW);

	return Vector4f(color / 255.0f, 255.0f);
}

#define ReadVoxelSDFDiscardZero(dst, pos) voxel = readVoxel(found, tsdf, pos, direction); if (voxel.w_depth == 0) return Vector3f(0, 0, 0); dst = voxel.sdf;

/**
 * Computes raw (unnormalized) gradient from neighboring voxels
 */
template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline Vector3f computeGradientFromSDF(
	const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3f& point, const TSDFDirection direction = TSDFDirection::NONE)
{
	bool found;

	TVoxel voxel;

	Vector3f ret(0, 0, 0);
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);
	Vector3f ncoeff(1.0f - coeff.x, 1.0f - coeff.y, 1.0f - coeff.z);

	// all 8 values are going to be reused several times
	Vector4f front, back;
	ReadVoxelSDFDiscardZero(front.x, pos + Vector3i(0, 0, 0));
	ReadVoxelSDFDiscardZero(front.y, pos + Vector3i(1, 0, 0));
	ReadVoxelSDFDiscardZero(front.z, pos + Vector3i(0, 1, 0));
	ReadVoxelSDFDiscardZero(front.w, pos + Vector3i(1, 1, 0));
	ReadVoxelSDFDiscardZero(back.x, pos + Vector3i(0, 0, 1));
	ReadVoxelSDFDiscardZero(back.y, pos + Vector3i(1, 0, 1));
	ReadVoxelSDFDiscardZero(back.z, pos + Vector3i(0, 1, 1));
	ReadVoxelSDFDiscardZero(back.w, pos + Vector3i(1, 1, 1));

	Vector4f tmp;
	float p1, p2, v1;
	// gradient x
	p1 = front.x * ncoeff.y * ncoeff.z +
	     front.z * coeff.y * ncoeff.z +
	     back.x * ncoeff.y * coeff.z +
	     back.z * coeff.y * coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(-1, 0, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(-1, 1, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(-1, 0, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(-1, 1, 1));
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y * coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y * coeff.z +
	     tmp.w * coeff.y * coeff.z;
	v1 = p1 * coeff.x + p2 * ncoeff.x;

	p1 = front.y * ncoeff.y * ncoeff.z +
	     front.w * coeff.y * ncoeff.z +
	     back.y * ncoeff.y * coeff.z +
	     back.w * coeff.y * coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(2, 0, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(2, 1, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(2, 0, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(2, 1, 1));
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y * coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y * coeff.z +
	     tmp.w * coeff.y * coeff.z;

	ret.x = TVoxel::valueToFloat(p1 * ncoeff.x + p2 * coeff.x - v1);

	// gradient y
	p1 = front.x * ncoeff.x * ncoeff.z +
	     front.y * coeff.x * ncoeff.z +
	     back.x * ncoeff.x * coeff.z +
	     back.y * coeff.x * coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, -1, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, -1, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, -1, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, -1, 1));
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y * coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x * coeff.z +
	     tmp.w * coeff.x * coeff.z;
	v1 = p1 * coeff.y + p2 * ncoeff.y;

	p1 = front.z * ncoeff.x * ncoeff.z +
	     front.w * coeff.x * ncoeff.z +
	     back.z * ncoeff.x * coeff.z +
	     back.w * coeff.x * coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, 2, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, 2, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, 2, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, 2, 1));
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y * coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x * coeff.z +
	     tmp.w * coeff.x * coeff.z;

	ret.y = TVoxel::valueToFloat(p1 * ncoeff.y + p2 * coeff.y - v1);

	// gradient z
	p1 = front.x * ncoeff.x * ncoeff.y +
	     front.y * coeff.x * ncoeff.y +
	     front.z * ncoeff.x * coeff.y +
	     front.w * coeff.x * coeff.y;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, 0, -1));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, 0, -1));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, 1, -1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, 1, -1));
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y * coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x * coeff.y +
	     tmp.w * coeff.x * coeff.y;
	v1 = p1 * coeff.z + p2 * ncoeff.z;

	p1 = back.x * ncoeff.x * ncoeff.y +
	     back.y * coeff.x * ncoeff.y +
	     back.z * ncoeff.x * coeff.y +
	     back.w * coeff.x * coeff.y;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, 0, 2));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, 0, 2));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, 1, 2));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, 1, 2));
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y * coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x * coeff.y +
	     tmp.w * coeff.x * coeff.y;

	ret.z = TVoxel::valueToFloat(p1 * ncoeff.z + p2 * coeff.z - v1);
	return ret;
}

/**
 * Computes raw (unnormalized) gradient from neighboring voxels Integer version (reduced overhead)
 */
template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline Vector3f computeGradientFromSDF(
	const Map<TIndex, TVoxel*, Args...>& tsdf, const Vector3i& pos, const TSDFDirection direction = TSDFDirection::NONE)
{
	bool found;
	TVoxel voxel;
	Vector3f ret(0, 0, 0);
	float p1, p2;

	// gradient x
	ReadVoxelSDFDiscardZero(p1, pos + Vector3i(1, 0, 0));
	ReadVoxelSDFDiscardZero(p2, pos + Vector3i(-1, 0, 0));
	ret.x = TVoxel::valueToFloat(p1 - p2);

	// gradient y
	ReadVoxelSDFDiscardZero(p1, pos + Vector3i(0, 1, 0));
	ReadVoxelSDFDiscardZero(p2, pos + Vector3i(0, -1, 0));
	ret.y = TVoxel::valueToFloat(p1 - p2);

	// gradient z
	ReadVoxelSDFDiscardZero(p1, pos + Vector3i(0, 0, 1));
	ReadVoxelSDFDiscardZero(p2, pos + Vector3i(0, 0, -1));
	ret.z = TVoxel::valueToFloat(p1 - p2);

	return ret;
}

/**
 * @param tau = voxelSize / truncationDistance (normalized value per 1 voxel). If 0 don't filter
 * @return
 */
template<typename T, typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline Vector3f computeSingleNormalFromSDF(
	const Map<TIndex, TVoxel*, Args...> tsdf, const ORUtils::Vector3<T>& point,
	const TSDFDirection direction = TSDFDirection::NONE, const float tau = 0.0)
{
	Vector3f gradient = computeGradientFromSDF(tsdf, point, direction);

	if (gradient == Vector3f(0, 0, 0))
		return gradient;

	if (tau > 0)
	{
		// Check each direction maximum 2 * truncationDistance / voxelSize + margin
		if (abs(gradient.x) > 2.4 * tau or abs(gradient.y) > 2.4 * tau or abs(gradient.z) > 2.4 * tau)
			return Vector3f(0, 0, 0);
		// Check, if gradient too unreliable (very close values in neighboring voxels). minimum expected length: (2 * tau)^2
		if (ORUtils::dot(gradient, gradient) < 2 * (tau * tau)) return Vector3f(0, 0, 0);
	}

	return gradient.normalised();
}

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(const Map<TIndex, TVoxel*, Args...> tsdf, const Vector3f& point)
{
	TVoxel voxel;
	Vector3f color(0.0f);
	float w_color = 0;
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);

	bool found;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0));
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0));
	color += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0));
	color += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0));
	color += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1));
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1));
	color += (coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1));
	color += (1.0f - coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1));
	color += (coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();

	return Vector4f(color / 255.0f, 255.0f);
}

//template<bool hasColor, typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
//struct VoxelColorReader;
//
//template<typename TVoxel, typename TIndex, template<typename, typename...> class Map, typename... Args>
//struct VoxelColorReader<false, TVoxel, TIndex, Map, Args> {
//	_CPU_AND_GPU_CODE_ static Vector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
//	                                               const THREADPTR(Vector3f) & point, const TSDFDirection direction)
//	{
//		return Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
//	}
//};
//
//template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
//struct VoxelColorReader<true, TVoxel, TIndex, Map, Args> {
//	_CPU_AND_GPU_CODE_ static Vector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
//	                                               const THREADPTR(Vector3f) & point, const TSDFDirection direction)
//	{
//		typename TIndex::IndexCache cache;
//		return readFromSDF_color4u_interpolated(voxelData, voxelIndex, point, direction, cache);
//	}
//};

} // namespace ITMlib