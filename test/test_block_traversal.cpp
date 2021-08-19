#include <test/catch2/catch.hpp>
#include <ITMLib/Utils/ITMMath.h>

#include "ITMLib/Engines/Reconstruction/Shared/ITMBlockTraversal.h"

// Maximum distance along ray
const float truncation_distance = 0.4;
const float block_size = 0.05;

using namespace ITMLib;

TEST_CASE("test basic functionality", "[block_traversal]")
{
  Vector3f origin(0, 0, 0);
  Vector3f direction(2, 0, 0);

  BlockTraversal block_traversal(
      origin,
      direction,
      truncation_distance,
      block_size
  );
  int count = 0;

  REQUIRE(block_traversal.HasNextBlock());

  // Check if first block is correct
  Vector3i block = block_traversal.GetNextBlock();
  count += 1;
  Vector3i origin_block(0, 0, 0);
  REQUIRE(block == origin_block);

  while (block_traversal.HasNextBlock())
  {
    block_traversal.GetNextBlock();
    count += 1;
  }
  REQUIRE(count == truncation_distance / block_size);
}

TEST_CASE("test non-axis-aligned direction", "[block_traversal]")
{
  Vector3f origin(1.01, 1.12, 1.23);
  Vector3f direction(-1, -1, -1); // 45 degrees to all axes

  BlockTraversal block_traversal(
      origin,
      direction,
      truncation_distance,
      block_size
  );

  // Check if internal parameters are correctly initialized
  REQUIRE(block_traversal.tMax.x * block_traversal.direction.x == Approx(-0.01f));
  REQUIRE(block_traversal.tMax.y * block_traversal.direction.y == Approx(-0.02f));
  REQUIRE(block_traversal.tMax.z * block_traversal.direction.z == Approx(-0.03f));

  // Check outcoming blocks
  Vector3i block(20, 22, 25); // 25 because 0.03 round up to next block!
  int count = 0;
  while (block_traversal.HasNextBlock())
  {
    REQUIRE(block_traversal.GetNextBlock() == block);

    // Update expected outcome for next turn (straight line, repeatedly crossing x,y,z in that order
    if (count % 3 == 0)
    {
      block.x -= 1;
    } else if (count % 3 == 1)
    {
      block.y -= 1;
    } else if (count % 3 == 2)
    {
      block.z -= 1;
    }
    count++;
  }
}

TEST_CASE("test truncation range", "[block_traversal]")
{
//  Vector3f origin = make_Vector3f(-0.303607, -0.516774, 0.001197);
//  Vector3f direction = make_Vector3f(-0.532084, -0.846631, -0.010123);
  Vector3f origin(-0.013223, 0.149254, 0.003236);
  Vector3f direction(0.111472, -0.993766, 0.001749);

  BlockTraversal block_traversal(
      origin,
      direction,
      truncation_distance,
      block_size
  );

  int count = 0;
  Vector3i block;
  while (block_traversal.HasNextBlock())
  {
    block = block_traversal.GetNextBlock();
//    printf("(%i, %i, %i), ", block.x, block.y, block.z);
    count += 1;
  }
  float dist = length(block.toFloat() * block_size - origin);
  REQUIRE(dist >= truncation_distance - block_size);
}

TEST_CASE("test start position rounding", "[block_traversal]")
{
  Vector3f origin(0.21f, -0.21f, 0.31f); // -> start position (1, -2, 1)
  Vector3f direction(0, 1, 0);

  BlockTraversal block_traversal(
      origin,
      direction,
      0.8, // truncation distance
      0.2, // block size
      false // switch off rounding to nearest responsible index
  );

  Vector3f p = block_traversal.WorldToBlockf(origin);
  Vector3f offset(SIGN(p.x) * 0.5f, SIGN(p.y) * 0.5f, SIGN(p.z) * 0.5f);
  Vector3i start_pos(floorf(p.x + offset.x), floorf(p.y + offset.y), floorf(p.z + offset.z));

  Vector3i block = block_traversal.GetNextBlock();
  REQUIRE(block == Vector3i(1, -2, 1)); // check if rounding is done correctly
}
