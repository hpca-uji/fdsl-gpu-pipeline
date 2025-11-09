from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    MixUp,
    CutMix,
    RandomChoice,
    RandomErasing,
    Normalize,
    InterpolationMode,
    RandomResizedCrop,
    ToDtype,
    Compose,
    InterpolationMode,
)
from torchvision.transforms.v2._auto_augment import RandAugment as RandAugmentV2
from torchvision.transforms.v2.functional._meta import get_size
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pycuda.driver as drv
from pycuda.compiler import DynamicSourceModule


_FILL = 128


pycuda_kernel_string = """
    #include <fstream>
    #include <curand_kernel.h>
    #include <cuda_fp16.h>

    #define _USE_MATH_DEFINES

    extern "C" {

        __device__ const unsigned char PERM[] = {
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
        36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
        234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
        88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71,
        134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
        230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
        1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
        116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250,
        124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227,
        47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44,
        154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98,
        108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
        242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14,
        239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121,
        50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243,
        141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131,
        13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37,
        240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252,
        219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125,
        136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158,
        231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245,
        40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187,
        208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198,
        173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126,
        255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223,
        183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167,
        43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185,
        112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179,
        162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199,
        106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236,
        205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156,
        180};

        // Linear interpolation macro
        #define lerp(t, a, b) ((a) + (t) * ((b) - (a)))

        // Gradient function
        __device__ inline float grad1(const int hash, const float x)
        {
            float g = (hash & 7) + 1.0f;
            if (hash & 8)
                g = -g;
            return (g * x);
        }

        // 1D Perlin noise function
        __device__ float noise1(float x, const int repeat, const int base)
        {
            float fx;
            int i = (int)floorf(x) % repeat;
            int ii = (i + 1) % repeat;
            i = (i & 255) + base;
            ii = (ii & 255) + base;

            x -= floorf(x);
            fx = x * x * x * (x * (x * 6 - 15) + 10);

            return lerp(fx, grad1(PERM[i], x), grad1(PERM[ii], x - 1)) * 0.4f;
        }

        // <<<NUM_BLOCKS=1, NUM_THREADS=norbits>>>
        __global__ void gen_color(uint8_t *__restrict colors, unsigned int seed, unsigned int init_seq)
        {
            int orbitidx = threadIdx.x;
            curandStatePhilox4_32_10_t state;
            curand_init(seed, init_seq + orbitidx, 0ULL, &state);
            uint8_t gray = roundf(curand_uniform(&state) * 255.0f);
            colors[orbitidx] = gray;
        }

        // <<<NUM_BLOCKS=norbits, NUM_THREADS=nvertex>>>
        __global__ void gen_noise(half2 *__restrict noise, unsigned int seed, unsigned int init_seq)
        {
            int vertexidx = threadIdx.x;
            int orbitidx = blockIdx.x;
            int noiseidx = blockDim.x * orbitidx + vertexidx;
            curandStatePhilox4_32_10_t state;
            curand_init(seed, init_seq + noiseidx, 0ULL, &state);
            float x = noise1(curand_uniform(&state) * 10000.0f, 1024, 0);
            float y = noise1(curand_uniform(&state) * 10000.0f, 1024, 0);
            float2 n = make_float2(x, y);
            noise[noiseidx] = __float22half2_rn(n);
        }
        // <<<NUM_BLOCKS=1, NUM_THREADS=1>>>
        __global__ void gen_pos(int2 *__restrict pos, const int res, unsigned int seed, unsigned int init_seq)
        {
            curandStatePhilox4_32_10_t state;
            curand_init(seed, init_seq, 0ULL, &state);
            int x = roundf(curand_uniform(&state) * res);
            int y = roundf(curand_uniform(&state) * res);
            pos[0] = make_int2(x, y);
        }

        // <<<NUM_BLOCKS=1, NUM_THREADS=1>>>
        __global__ void rng(half2 *__restrict noise, uint8_t *__restrict colors,
                                int2 *__restrict pos, const int nvertex, const int norbits,
                                const int res, unsigned int seed)
        {

            // Compute totalvertex
            int totalvertex = norbits * nvertex;

            // Generate random noise
            int NUM_THREADS = nvertex;
            int NUM_BLOCKS = norbits;
            gen_noise<<<NUM_BLOCKS, NUM_THREADS, 0, cudaStreamFireAndForget>>>(noise, seed, 0);

            // Generate random colors
            NUM_THREADS = norbits;
            NUM_BLOCKS = 1;
            gen_color<<<NUM_BLOCKS, NUM_THREADS, 0, cudaStreamFireAndForget>>>(colors, seed, totalvertex);

            // Generate random pos
            NUM_THREADS = 1;
            NUM_BLOCKS = 1;
            gen_pos<<<NUM_BLOCKS, NUM_THREADS, 0, cudaStreamFireAndForget>>>(pos, res, seed, totalvertex + norbits);
        }

        // <<<NUM_BLOCKS=1, NUM_THREADS=nvertex>>>
        __global__ void compute_vertex(float2 *__restrict vertex,
                                    const half2 *__restrict noise,
                                    const int2 *__restrict pos, const int nvertex,
                                    const int norbits, const float ovalx, const float ovaly,
                                    const int freq1, const int freq2, const float noisecoef,
                                    const int startrad, const float linewidth, const int res)
        {

            // Compute idx
            int vertexidx = threadIdx.x;

            // Compute param angle
            float angle = (M_PI * 2) / nvertex;
            float paramangle = angle * vertexidx;
            float cosangle = cosf(paramangle);
            float sinangle = sinf(paramangle);

            // Compute base wave
            float wave = 0.5 * sinf(paramangle * freq1) + 0.5 * sinf(paramangle * freq2);

            // Compute initial vertex
            vertex[vertexidx].x = cosangle * startrad * ovalx + pos[0].x;
            vertex[vertexidx].y = sinangle * startrad * ovaly + pos[0].y;

            // Compute noise for initial vertex
            float2 noise_f = __half22float2(noise[vertexidx]);
            float noise_x = noise_f.x * noisecoef - noisecoef - wave;
            float noise_y = noise_f.y * noisecoef - noisecoef - wave;

            // Modify initial vertex
            vertex[vertexidx].x -= (cosangle * (noise_x - linewidth));
            vertex[vertexidx].y -= (sinangle * (noise_y - linewidth));

            // Prepare noise loop foreach orbit
            int id, prev_id;
            prev_id = vertexidx;

            // Loop over each orbit
            for (int i = 1; i < norbits; i++)
            {
                // Compute access id
                id = nvertex * i + vertexidx;

                // Compute noise
                noise_f = __half22float2(noise[id]);
                noise_x = noise_f.x * noisecoef - noisecoef - wave;
                noise_y = noise_f.y * noisecoef - noisecoef - wave;

                // Create new vertex from previous vertex + noise
                vertex[id].x = vertex[prev_id].x - (cosangle * (noise_x - linewidth));
                vertex[id].y = vertex[prev_id].y - (sinangle * (noise_y - linewidth));

                // Refresh prev_id with current access id
                prev_id = id;
            }
        }

        // Draw line (bresenham algo)
        __device__ void draw_line(uint8_t *__restrict imgarr, int x0, int y0, int x1, int y1, int res, uint8_t gray)
        {

            int i, n, e;
            int dx, dy;
            int xs, ys;

            // Check if both points are out of bounds
            if ((x0 < 0 || x0 >= res || y0 < 0 || y0 >= res) && (x1 < 0 || x1 >= res || y1 < 0 || y1 >= res))
            {
                return;
            }

            /* normalize coordinates */
            dx = x1 - x0;
            if (dx < 0)
            {
                dx = -dx, xs = -1;
            }
            else
            {
                xs = 1;
            }
            dy = y1 - y0;
            if (dy < 0)
            {
                dy = -dy, ys = -1;
            }
            else
            {
                ys = 1;
            }

            n = (dx > dy) ? dx : dy;

            if (dx == 0)
            {
                /* vertical */
                for (i = 0; i < dy; i++)
                {
                    if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                    {
                        imgarr[y0 * res + x0] = gray;
                    }
                    y0 += ys;
                }
            }
            else if (dy == 0)
            {
                /* horizontal */
                for (i = 0; i < dx; i++)
                {
                    if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                    {
                        imgarr[y0 * res + x0] = gray;
                    }
                    x0 += xs;
                }
            }
            else if (dx > dy)
            {
                /* bresenham, horizontal slope */
                n = dx;
                dy += dy;
                e = dy - dx;
                dx += dx;

                for (i = 0; i < n; i++)
                {
                    if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                    {
                        imgarr[y0 * res + x0] = gray;
                    }
                    if (e >= 0)
                    {
                        y0 += ys;
                        e -= dx;
                    }
                    e += dy;
                    x0 += xs;
                }
            }
            else
            {
                /* bresenham, vertical slope */
                n = dy;
                dx += dx;
                e = dx - dy;
                dy += dy;

                for (i = 0; i < n; i++)
                {
                    if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                    {
                        imgarr[y0 * res + x0] = gray;
                    }
                    if (e >= 0)
                    {
                        x0 += xs;
                        e -= dy;
                    }
                    e += dx;
                    y0 += ys;
                }
            }
        }

        // <<<NUM_BLOCKS=*, NUM_THREADS=(32,8)>>>
        __global__ void draw_vertex(uint8_t *__restrict imgarr, const float2 *__restrict vertex,
                                    const uint8_t *__restrict colors,
                                    int nvertex, int norbits, int res)
        {

            // Compute access idx
            int vertexidx = blockIdx.x * blockDim.x + threadIdx.x;
            int orbitidx = blockIdx.y * blockDim.y + threadIdx.y;
            if (vertexidx >= nvertex || orbitidx >= norbits)
                return;
            int access_idx = orbitidx * nvertex + vertexidx;

            // Get coordinates of origin vertex of the line and cast to int
            int x0 = (int)vertex[access_idx].x;
            int y0 = (int)vertex[access_idx].y;

            // Get coordinates of final vertex of the line and cast to int
            int x1, y1;
            if (vertexidx == (nvertex - 1))
            {
                // If origin vertex is the last vertex of the line, connect with first
                int first_access_idx = orbitidx * nvertex;
                x1 = (int)vertex[first_access_idx].x;
                y1 = (int)vertex[first_access_idx].y;
            }
            else
            {
                x1 = (int)vertex[access_idx + 1].x;
                y1 = (int)vertex[access_idx + 1].y;
            }

            // Draw line with bresenham
            draw_line(imgarr, x0, y0, x1, y1, res, colors[orbitidx]);
        }

        __global__ void genvatom_image(uint8_t *__restrict imgarr, float2 *__restrict vertex,
                                        half2 *__restrict noise, uint8_t *__restrict colors,
                                        int2 *__restrict pos, int nvertex, int norbits, float ovalx, float ovaly,
                                        int freq1, int freq2, float noisecoef, int startrad, float linewidth, int res, int seed)
        {

            // Generate random data using subkernels with cudaStreamFireAndForget
            rng<<<1, 1>>>(noise, colors, pos, nvertex, norbits, res, seed);

            // Compute vertex
            int NUM_THREADS = nvertex;
            int NUM_BLOCKS = 1;
            // Launching compute_vertex
            compute_vertex<<<NUM_BLOCKS, NUM_THREADS>>>(vertex, noise, pos, nvertex, norbits, ovalx, ovaly,
                                                        freq1, freq2, noisecoef, startrad, linewidth, res);

            // Draw vertex
            dim3 block(32, 8);
            dim3 grid((nvertex + 32 - 1) / 32, (norbits + 8 - 1) / 8);
            draw_vertex<<<grid, block>>>(imgarr, vertex, colors, nvertex, norbits, res);
        }

        __global__ void genvatom_tail_marker(){
            return;
        }

        // <<<NUM_BLOCKS=1, NUM_THREADS=batch_size>>>
        __global__ void genvatom_batch(uint8_t *d_imgarr_base, float2 *d_vertex_base,
                                                half2 *d_noise_base, uint8_t *d_colors_base,
                                                int2 *d_pos_base, int maxtotalvertex, int norbits_max,
                                                int *nvertex, int *norbits, float *ovalx,
                                                float *ovaly, int *freq1, int *freq2, float *noisecoef,
                                                int *startrad, float *linewidth, int res,
                                                int base_seed, int batch_size, int *classes_idx)
        {
            // The index of the image to generate
            int img_idx = threadIdx.x;

            // Calculate the offsets for the current image's data buffers
            // Sizes for one image
            const size_t img_offset = (size_t)img_idx * res * res;
            const size_t vertex_offset = (size_t)img_idx * maxtotalvertex;
            const size_t colors_offset = (size_t)img_idx * norbits_max;
            const size_t pos_offset = (size_t)img_idx;

            // Pointers for the current image's data
            uint8_t *imgarr_ptr = d_imgarr_base + img_offset;
            float2 *vertex_ptr = d_vertex_base + vertex_offset;
            half2 *noise_ptr = d_noise_base + vertex_offset;
            uint8_t *colors_ptr = d_colors_base + colors_offset;
            int2 *pos_ptr = d_pos_base + pos_offset;

            // Use a unique seed for each image
            unsigned int current_seed = base_seed * batch_size + img_idx;
            int class_idx = classes_idx[img_idx];

            genvatom_image<<<1, 1, 0, cudaStreamFireAndForget>>>(imgarr_ptr, vertex_ptr, noise_ptr, colors_ptr,
                                                    pos_ptr, nvertex[class_idx], norbits[class_idx],
                                                    ovalx[class_idx], ovaly[class_idx], freq1[class_idx],
                                                    freq2[class_idx], noisecoef[class_idx], startrad[class_idx],
                                                    linewidth[class_idx], res, current_seed);
        }
    }
"""


class Holder(drv.PointerHolderBase):
    """Class to use torch Tensors on cuda kernel"""

    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class RandAugmentV2WithInvert(RandAugmentV2):

    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "ShearY": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins),
            True,
        ),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(
                0.0, 150.0 / 331.0 * width, num_bins
            ),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(
                0.0, 150.0 / 331.0 * height, num_bins
            ),
            True,
        ),
        "Rotate": (
            lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins),
            True,
        ),
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Posterize": (
            lambda num_bins, height, width: (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4))
            )
            .round()
            .int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
        "Invert": (lambda num_bins, height, width: None, False),
    }

    def forward(self, *inputs: Any) -> Any:
        flat_inputs_with_spec, image_or_video = (
            self._flatten_and_extract_image_or_video(inputs)
        )
        height, width = get_size(image_or_video)  # type: ignore[arg-type]

        for _ in range(self.num_ops):
            transform_id, (magnitudes_fn, signed) = self._get_random_item(
                self._AUGMENTATION_SPACE
            )
            magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
            if (
                magnitudes is not None and torch.rand(()) <= 0.5
            ):  # Add torch.rand to mimic timm (0.5 prob of appplying transform)
                magnitude = float(magnitudes[self.magnitude])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0
            image_or_video = self._apply_image_or_video_transform(
                image_or_video,
                transform_id,
                magnitude,
                interpolation=self.interpolation,
                fill=self._fill,
            )

        return self._unflatten_and_insert_image_or_video(
            flat_inputs_with_spec, image_or_video
        )


class VatomPrefetchLoader:
    """
    Prefetch loader that:
      - compiles a CUDA kernel from `pycuda_kernel_string` in __init__
      - allocates reusable buffers
      - in __iter__ enqueues per-sample kernel launches into a dedicated stream that write into a
        preallocated batch tensor (out_batch).
      - optionally runs torchvision v2 batch transforms on the GPU
    """

    def __init__(
        self,
        loader: DataLoader,
        device: int,
        batch_size: int,
        classes: list,
        res: int = 224,
        nclasses: int = 1000,
        norbits_max: int = 200,
        nvertex_max: int = 1000,
        num_ops: int = 2,
        magnitude: int = 9,
        mixup: float = 0.5,
        cutmix: float = 0.5,
        label_smoothing: float = 0.1,
        reprob: float = 0.25,
    ):
        """
        loader: DataLoader that yields batches of (idxs_cpu_tensor, class_ids_cpu_tensor)
        device: torch.device('cuda:0') for example
        pycuda_kernel_string: full CUDA C source as a Python string
        kernel_function_name: name of function inside that module to call
        classes: list-of-tensor with per-class scalars for kernel (host-side)
        res: image resolution (H==W==res)
        norbits_max, nvertex_max: sizes used to allocate scratch buffers
        """
        self.loader = loader
        self.device = device
        self.batch_size = batch_size
        self.res = res
        self.classes = classes
        self.nclasses = nclasses
        self.label_smoothing = label_smoothing
        self.norbits_max = norbits_max
        self.totalvertex_max = norbits_max * nvertex_max

        # --- Initialize pycuda and push context for this device (one-time) ---
        drv.init()
        self.ctx = drv.Device(device).retain_primary_context()
        self.ctx.push()

        # --- compile module from string once and store function handle ---
        self.mod = DynamicSourceModule(
            source=pycuda_kernel_string,
            options=["-rdc=true", "-lcudadevrt"],
            no_extern_c=True,
        )
        self.genvatom = self.mod.get_function("genvatom_batch")

        # --- allocate per-prefetcher reusable device buffers (persistent) ---
        # Adjust types/sizes to match your kernel expectations
        self.colors = torch.empty(
            [self.batch_size * norbits_max], dtype=torch.uint8, device=self.device
        )
        self.vertex = torch.empty(
            [self.batch_size * self.totalvertex_max, 2],
            dtype=torch.float32,
            device=self.device,
        )
        self.noise = torch.empty(
            [self.batch_size * self.totalvertex_max, 2],
            dtype=torch.float16,
            device=self.device,
        )
        self.pos = torch.empty(
            [self.batch_size, 2], dtype=torch.float32, device=self.device
        )
        self.prepare_classes_pointers()

        # Set up pre transforms
        if num_ops == 0 or magnitude == 0:
            self.pre_transform = Compose(
                [
                    RandomResizedCrop(
                        size=(res, res), interpolation=InterpolationMode.BICUBIC
                    ),
                    ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.pre_transform = None
            self.resize = RandomResizedCrop(
                size=(res, res), interpolation=InterpolationMode.BICUBIC
            )
            self.ra = RandAugmentV2WithInvert(
                num_ops=num_ops, magnitude=magnitude, fill=_FILL
            )
            self.todtype = ToDtype(torch.float32, scale=True)

        # Define mixup/cutmix
        self.mix = None

        # Set up mixup
        if mixup > 0.0:
            mixup_fn = MixUp(alpha=mixup, num_classes=nclasses)
            self.mix = mixup_fn

        # Set up cutmix
        if cutmix > 0.0:
            cutmix_fn = CutMix(alpha=cutmix, num_classes=nclasses)
            self.mix = cutmix_fn

        # If both active, create RandomChoice
        if mixup > 0.0 and cutmix > 0.0:

            self.mix = RandomChoice([mixup_fn, cutmix_fn])

        # Create norm layer
        self.norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        if reprob > 0:
            self.re = RandomErasing(p=reprob, value="random")
        else:
            self.re = None

    def prepare_classes_pointers(self):

        for i in range(len(self.classes)):
            self.classes[i] = Holder(self.classes[i])

    def clear_ctx(self):
        self.ctx.pop()
        self.ctx.detach()

    def __iter__(self):
        """
        Underlying loader must yield batches of (idxs_cpu_tensor, class_ids_cpu_tensor), both on CPU.
        Example collate: collate idxs/class ids into 1D CPU tensors with shape (B,)
        """

        first = True

        # 1. Create PyCUDA stream
        pycuda_stream = drv.Stream()

        # 2. Wrap it with a PyTorch external stream
        torch_generation_stream = torch.cuda.ExternalStream(pycuda_stream.handle)

        # 3. Create a second PyTorch stream for transforms
        torch_transform_stream = torch.cuda.Stream()

        # 4. Create an event to signal completion
        kernel_done_event = torch.cuda.Event(blocking=False)

        for idxs, labels in self.loader:

            # Enqueue kernel launches + aug on generation stream
            with torch.cuda.stream(torch_generation_stream):

                # preallocate device batch output buffer once per batch
                # kernel writes a single-channel uint8 image per sample; we keep (B,1,H,W)
                out = torch.zeros(
                    (self.batch_size, 1, self.res, self.res),
                    dtype=torch.uint8,
                    device=self.device,
                )

                # Move labels to device (non_blocking) - they might be converted to soft labels by MixUp
                labels = labels[0]
                labels = labels.to(self.device, non_blocking=True)

                # Convert idxs to scalar
                idxs = idxs.item()

                # Create kernel args
                args = [
                    Holder(out),
                    Holder(self.vertex),
                    Holder(self.noise),
                    Holder(self.colors),
                    Holder(self.pos),
                    np.int32(self.totalvertex_max),
                    np.int32(self.norbits_max),
                    *self.classes,
                    np.int32(self.res),
                    np.int32(idxs),
                    np.int32(self.batch_size),
                    Holder(labels),
                ]

                # Launch kernel (grid/block values are placeholders—replace with your kernel's choices)
                self.genvatom(
                    *args,
                    grid=(1, 1, 1),
                    block=(self.batch_size, 1, 1),
                    stream=pycuda_stream
                )

                # Register event
                kernel_done_event.record(torch_generation_stream)

            with torch.cuda.stream(torch_transform_stream):

                # Wait for event
                torch_transform_stream.wait_event(kernel_done_event)

                # Expand to 3 channels
                out = out.repeat(1, 3, 1, 1)

                # Detect pre transforms
                if self.pre_transform:
                    out = self.pre_transform(out)
                else:
                    out = self.resize(out)  # Batched resize
                    for i in range(
                        self.batch_size
                    ):  # Sequential RA cause batched execution applies same transform to all imgages
                        out[i] = self.ra(out[i])
                    out = self.todtype(out)  # Batched todtype

                # Apply mixup
                if self.mix is not None:

                    out, labels = self.mix(out, labels)

                    labels = (
                        labels * (1.0 - self.label_smoothing)
                        + self.label_smoothing / self.nclasses
                    )
                else:
                    labels = torch.nn.functional.one_hot(labels, self.nclasses)

                # Apply batched norm
                out = self.norm(out)

                # Apply sequential randomerasing for same reason as RA
                if self.re:
                    for i in range(self.batch_size):
                        out[i] = self.re(out[i])

            # Yield one batch behind (prefetch pattern)
            if not first:
                yield prev_images, prev_labels
            else:
                first = False

            # Ensure main stream waits for generation stream so the returned tensors are ready
            torch.cuda.current_stream(device=self.device).wait_stream(
                torch_transform_stream
            )

            # Save prepared batch for next yield
            prev_images = out
            prev_labels = labels

        # yield last prepared batch (if any)
        yield prev_images, prev_labels

    def __len__(self):
        return len(self.loader)
