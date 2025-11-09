#include <vector>
#include <fstream>
#include <curand_kernel.h>

#define _USE_MATH_DEFINES

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

// <<<NUM_BLOCKS=norbits, NUM_THREADS=nvertex>>>
__global__ void gen_noise(float2 *__restrict noise, unsigned int seed, unsigned int init_seq)
{
    int vertexidx = threadIdx.x;
    int orbitidx = blockIdx.x;
    int noiseidx = blockDim.x * orbitidx + vertexidx;
    curandState state;
    curand_init(seed, init_seq + noiseidx, 0ULL, &state);
    float x = noise1(curand_uniform(&state) * 10000.0f, 1024, 0);
    float y = noise1(curand_uniform(&state) * 10000.0f, 1024, 0);
    noise[noiseidx] = make_float2(x, y);
}
// <<<NUM_BLOCKS=1, NUM_THREADS=1>>>
__global__ void gen_pos(int2 *__restrict pos, const int res, unsigned int seed, unsigned int init_seq)
{
    curandState state;
    curand_init(seed, init_seq, 0ULL, &state);
    int x = roundf(curand_uniform(&state) * res);
    int y = roundf(curand_uniform(&state) * res);
    pos[0] = make_int2(x, y);
}

// <<<NUM_BLOCKS=1, NUM_THREADS=1>>>
__global__ void rng(float2 *__restrict noise, uint8_t *__restrict colors,
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
                               const float2 *__restrict noise,
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
    float noise_x = noise[vertexidx].x * noisecoef - noisecoef - wave;
    float noise_y = noise[vertexidx].y * noisecoef - noisecoef - wave;

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
        noise_x = noise[id].x * noisecoef - noisecoef - wave;
        noise_y = noise[id].y * noisecoef - noisecoef - wave;

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

void save_image(std::vector<uint8_t> &imgarr, int res)
{
    FILE *f = fopen("img.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", res, res);
    for (int y = 0; y < res; y++)
    {
        for (int x = 0; x < res; x++)
        {
            int value = imgarr[y * res + x];
            fputc(value, f); // 0 .. 255
            fputc(value, f); // 0 .. 255
            fputc(value, f); // 0 .. 255
        }
    }
    fclose(f);
}

int main()
{

    // Static vars //
    constexpr int nvertex = 1000;
    constexpr int norbits = 200;
    constexpr float ovalx = 1.2f;
    constexpr float ovaly = 1.8f;
    constexpr int freq1 = 5;
    constexpr int freq2 = 8;
    constexpr float noisecoef = 2.0f;
    constexpr int startrad = 25;
    constexpr float linewidth = 0.01f;
    constexpr int res = 224;
    constexpr int seed = 42;

    /// Data setup

    // Compute sizes for arrays
    constexpr int totalvertex = nvertex * norbits;
    constexpr size_t imgsize = sizeof(uint8_t) * res * res;
    constexpr size_t vertexsize = sizeof(float2) * totalvertex;
    constexpr size_t colorsize = sizeof(uint8_t) * norbits;
    constexpr size_t posize = sizeof(int2);

    // Create CPU array for img
    std::vector<uint8_t> imgarr(res * res, 0);

    // Pointers to arrays on device
    uint8_t *d_imgarr, *d_colors;
    int2 *d_pos;
    float2 *d_vertex, *d_noise;

    // Reserve space on device
    cudaMalloc(&d_imgarr, imgsize);
    cudaMemset(d_imgarr, 0, imgsize);
    cudaMalloc(&d_vertex, vertexsize);
    cudaMalloc(&d_noise, vertexsize);
    cudaMalloc(&d_colors, colorsize);
    cudaMalloc(&d_pos, posize);

    // gen random
    rng<<<1, 1>>>(d_noise, d_colors, d_pos, nvertex, norbits, res, seed);

    // Compute vertex
    int NUM_THREADS = nvertex;
    int NUM_BLOCKS = 1;
    compute_vertex<<<NUM_BLOCKS, NUM_THREADS>>>(d_vertex, d_noise, d_pos, nvertex, norbits, ovalx, ovaly,
                                                freq1, freq2, noisecoef, startrad, linewidth, res);

    // Draw vertex
    dim3 block(32, 8);
    dim3 grid((nvertex + 32 - 1) / 32, (norbits + 8 - 1) / 8);
    draw_vertex<<<grid, block>>>(d_imgarr, d_vertex, d_colors, nvertex, norbits, res);

    // Get result and save to ppm file
    cudaMemcpy(imgarr.data(), d_imgarr, imgsize, cudaMemcpyDeviceToHost);
    save_image(imgarr, res);

    // Free resources
    cudaFree(d_imgarr);
    cudaFree(d_colors);
    cudaFree(d_vertex);
    cudaFree(d_noise);
    cudaFree(d_pos);

    return 0;
}