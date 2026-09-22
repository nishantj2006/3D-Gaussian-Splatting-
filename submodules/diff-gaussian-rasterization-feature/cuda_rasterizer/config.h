/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
// Upper bound for the per-Gaussian descriptor. The active channel count is
// inferred from the input tensor at runtime. 128D is the Orin profile used by
// this repository, while keeping smaller descriptors usable with --speedup.
#define MAX_SEMANTIC_CHANNELS 128

#define BLOCK_X 16
#define BLOCK_Y 16

#endif
