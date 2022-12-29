//
// Created by Malte Splietker on 04.06.19.
//

#pragma once

#define SDF_BLOCK_SIZE 8        // SDF block size
#define SDF_BLOCK_SIZE3 (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)

// Set to
// 0 render using fused colors
// 1 render colors according to direction weight (debug)
#define RENDER_DIRECTION_COLORS 0

#define MAXFLOAT ((float)3.40282346638528860e+38)