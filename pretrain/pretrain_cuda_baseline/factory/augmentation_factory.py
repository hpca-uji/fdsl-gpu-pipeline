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
            curandState state;
            curand_init(seed, init_seq + orbitidx, 0ULL, &state);
            uint8_t gray = roundf(curand_uniform(&state) * 255.0f);
            colors[orbitidx] = gray;
        }

        // <<<NUM_BLOCKS=norbits, NUM_THREADS=nvertex*2>>>
        __global__ void gen_noise(float *__restrict noise, unsigned int seed, unsigned int init_seq)
        {
            int vertexidx = threadIdx.x;
            int orbitidx = blockIdx.x;
            int noiseidx = blockDim.x * orbitidx + vertexidx;
            curandState state;
            curand_init(seed, init_seq + noiseidx, 0ULL, &state);
            noise[noiseidx] = noise1(curand_uniform(&state) * 10000.0f, 1024, 0);
        }

        // <<<NUM_BLOCKS=1, NUM_THREADS=1>>>
        __global__ void gen_pos(int *__restrict pos, const int res, unsigned int seed, unsigned int init_seq)
        {
            curandState state;
            curand_init(seed, init_seq, 0ULL, &state);
            pos[0] = roundf(curand_uniform(&state) * res);
            pos[1] = roundf(curand_uniform(&state) * res);
        }

        // <<<NUM_BLOCKS=1, NUM_THREADS=1>>>
        __global__ void rng(float *__restrict nx, float *__restrict ny, uint8_t *__restrict colors,
                                int *__restrict pos, const int nvertex, const int norbits,
                                const int res, unsigned int seed)
        {

            // Compute totalvertex
            int totalvertex = norbits * nvertex;

            // Generate random noise
            int NUM_THREADS = nvertex;
            int NUM_BLOCKS = norbits;
            gen_noise<<<NUM_BLOCKS, NUM_THREADS, 0, cudaStreamFireAndForget>>>(nx, seed, 0);
            gen_noise<<<NUM_BLOCKS, NUM_THREADS, 0, cudaStreamFireAndForget>>>(ny, seed, totalvertex);

            // Generate random colors
            NUM_THREADS = norbits;
            NUM_BLOCKS = 1;
            gen_color<<<NUM_BLOCKS, NUM_THREADS, 0, cudaStreamFireAndForget>>>(colors, seed, 2 * totalvertex);

            // Generate random pos
            NUM_THREADS = 1;
            NUM_BLOCKS = 1;
            gen_pos<<<NUM_BLOCKS, NUM_THREADS, 0, cudaStreamFireAndForget>>>(pos, res, seed, 2 * totalvertex + norbits);
        }

        // <<<NUM_BLOCKS=1, NUM_THREADS=nvertex>>>
        __global__ void compute_vertex(float *__restrict vx, float *__restrict vy,
                                    const float *__restrict nx, const float *__restrict ny,
                                    const int *__restrict pos, const int nvertex,
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
            vx[vertexidx] = cosangle * startrad * ovalx + pos[0];
            vy[vertexidx] = sinangle * startrad * ovaly + pos[1];

            // Compute noise for initial vertex
            float noise_x = nx[vertexidx] * noisecoef - noisecoef - wave;
            float noise_y = ny[vertexidx] * noisecoef - noisecoef - wave;

            // Modify initial vertex
            vx[vertexidx] -= (cosangle * (noise_x - linewidth));
            vy[vertexidx] -= (sinangle * (noise_y - linewidth));

            // Prepare noise loop foreach orbit
            int id, prev_id;
            prev_id = vertexidx;

            // Loop over each orbit
            for (int i = 1; i < norbits; i++)
            {
                // Compute access id
                id = nvertex * i + vertexidx;

                // Compute noise
                noise_x = nx[id] * noisecoef - noisecoef - wave;
                noise_y = ny[id] * noisecoef - noisecoef - wave;

                // Create new vertex from previous vertex + noise
                vx[id] = vx[prev_id] - (cosangle * (noise_x - linewidth));
                vy[id] = vy[prev_id] - (sinangle * (noise_y - linewidth));

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
        __global__ void draw_vertex(uint8_t *__restrict imgarr, const float *__restrict vx,
                                    const float *__restrict vy, const uint8_t *__restrict colors,
                                    int nvertex, int norbits, int res)
        {

            // Compute access idx
            int vertexidx = blockIdx.x * blockDim.x + threadIdx.x;
            int orbitidx = blockIdx.y * blockDim.y + threadIdx.y;
            if (vertexidx >= nvertex || orbitidx >= norbits)
                return;
            int access_idx = orbitidx * nvertex + vertexidx;

            // Get coordinates of origin vertex of the line and cast to int
            int x0 = (int)vx[access_idx];
            int y0 = (int)vy[access_idx];

            // Get coordinates of final vertex of the line and cast to int
            int x1, y1;
            if (vertexidx == (nvertex - 1))
            {
                // If origin vertex is the last vertex of the line, connect with first
                int first_access_idx = orbitidx * nvertex;
                x1 = (int)vx[first_access_idx];
                y1 = (int)vy[first_access_idx];
            }
            else
            {
                x1 = (int)vx[access_idx + 1];
                y1 = (int)vy[access_idx + 1];
            }

            // Draw line with bresenham
            draw_line(imgarr, x0, y0, x1, y1, res, colors[orbitidx]);
        }

        __global__ void genvatom_image(uint8_t *__restrict imgarr, float *__restrict vx, float *__restrict vy,
                                        float *__restrict nx, float *__restrict ny, uint8_t *__restrict colors,
                                        int *__restrict pos, int nvertex, int norbits, float ovalx, float ovaly,
                                        int freq1, int freq2, float noisecoef, int startrad, float linewidth, int res, int seed)
        {

            // gen random
            rng<<<1, 1>>>(nx, ny, colors, pos, nvertex, norbits, res, seed);

            // Compute vertex
            int NUM_THREADS = nvertex;
            int NUM_BLOCKS = 1;
            compute_vertex<<<NUM_BLOCKS, NUM_THREADS>>>(vx, vy, nx, ny, pos, nvertex, norbits, ovalx, ovaly,
                                                        freq1, freq2, noisecoef, startrad, linewidth, res);

            // Draw vertex
            dim3 block(32, 8);
            dim3 grid((nvertex + 32 - 1) / 32, (norbits + 8 - 1) / 8);
            draw_vertex<<<grid, block>>>(imgarr, vx, vy, colors, nvertex, norbits, res);
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
        classes: optional list-of-lists with per-class scalars for kernel (host-side)
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
        totalvertex_max = norbits_max * nvertex_max

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
        self.genvatom = self.mod.get_function("genvatom_image")

        # --- allocate per-prefetcher reusable device buffers (persistent) ---
        # Adjust types/sizes to match your kernel expectations
        self.colors = torch.empty([norbits_max], dtype=torch.uint8, device=self.device)
        self.vx = torch.empty(
            [totalvertex_max], dtype=torch.float32, device=self.device
        )
        self.vy = torch.empty(
            [totalvertex_max], dtype=torch.float32, device=self.device
        )
        self.nx = torch.empty(
            [totalvertex_max], dtype=torch.float32, device=self.device
        )
        self.ny = torch.empty(
            [totalvertex_max], dtype=torch.float32, device=self.device
        )
        self.pos = torch.empty([2], dtype=torch.float32, device=self.device)

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

    def clear_ctx(self):
        self.ctx.pop()
        self.ctx.detach()

    def generate_image(
        self, out_slice: torch.Tensor, idx: int, class_id: int, pycuda_stream
    ):
        """
        Launch the compiled kernel to write a single image into `out_slice` (shape [H,W] or [1,H,W]).
        Uses Holder to pass torch pointer into PyCUDA.
        IMPORTANT: do NOT call any host-synchronizing operations here.
        """
        # Build argument list similar to your earlier code. For this minimal example,
        # we pass: Holder(image), Holder(vx), Holder(vy), Holder(nx), Holder(ny),
        # Holder(colors), Holder(pos), class params..., res, idx, class_id
        args = [
            Holder(out_slice),
            Holder(self.vx),
            Holder(self.vy),
            Holder(self.nx),
            Holder(self.ny),
            Holder(self.colors),
            Holder(self.pos),
            *self.classes[class_id],
            np.int32(self.res),
            np.int32(idx),
        ]

        # Launch kernel (grid/block values are placeholders—replace with your kernel's choices)
        self.genvatom(*args, grid=(1, 1, 1), block=(1, 1, 1), stream=pycuda_stream)
        # No host synchronization here; kernel is enqueued to the current CUDA stream.

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

        for idxs, class_ids in self.loader:

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
                labels = class_ids.to(self.device, non_blocking=True)

                # Launch kernel per sample into out_batch[i,0]
                for i in range(self.batch_size):
                    idx = int(idxs[i].item())
                    class_id = int(class_ids[i].item())
                    # pass the slice (view) pointing into device memory
                    self.generate_image(out[i, 0], idx, class_id, pycuda_stream)

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
                    ):  # Sequential RA cause batched execution applies same transform to all img
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
