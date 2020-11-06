//
// Created by Malte Splietker on 04.06.19.
//

#pragma once

#define SDF_BLOCK_SIZE 8				// SDF block size
#define SDF_BLOCK_SIZE3 (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)

//#define SDF_LOCAL_BLOCK_NUM 0x40000		// Number of locally stored blocks, currently 2^18
//#define SDF_BUCKET_NUM 0x100000       // Number of Hash Bucket, should be 2^n and bigger than SDF_LOCAL_BLOCK_NUM
//#define SDF_EXCESS_LIST_SIZE 0x20000  // 0x20000 Size of excess list, used to handle collisions. Also max offset (unsigned short) value.

////// for loop closure
#define SDF_LOCAL_BLOCK_NUM 0x10000		// Number of locally stored blocks, currently 2^16
#define SDF_BUCKET_NUM 0x40000        // Number of Hash Bucket, should be 2^n and bigger than SDF_LOCAL_BLOCK_NUM
#define SDF_EXCESS_LIST_SIZE 0x8000   // 0x8000 Size of excess list, used to handle collisions. Also max offset (unsigned short) value.

//// for directional loop closure
//#define SDF_LOCAL_BLOCK_NUM 0x20000		// Number of locally stored blocks, currently 2^17
//#define SDF_BUCKET_NUM 0x80000        // Number of Hash Bucket, should be 2^n and bigger than SDF_LOCAL_BLOCK_NUM
//#define SDF_EXCESS_LIST_SIZE 0x10000	// 0x8000 Size of excess list, used to handle collisions. Also max offset (unsigned short) value.

#define SDF_HASH_MASK (SDF_BUCKET_NUM - 1)  // Used for get hashing value of the bucket index
#define SDF_TRANSFER_BLOCK_NUM 0x1000       // Maximum number of blocks transfered in one swap operation

// Set to 1 for combination between 6 independent renderings and 0 for combined raytracing (slower)
#define DIRECTIONAL_RENDERING_MODE 0