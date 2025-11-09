import random
import math
import time

from PIL import Image, ImageDraw
import noise
import numpy as np


class OriSyntDatasetClass:
    def __init__(
        self,
        image_size,
        vertex_number,
        perlin_noise_coefficient,
        line_width,
        start_rad,
        line_draw_num,
        nami1,
        nami2,
        oval_rate_x,
        oval_rate_y,
    ):

        self.image_size = image_size
        self.vertex_number = vertex_number
        self.perlin_noise_coefficient = perlin_noise_coefficient
        self.line_width = line_width
        self.start_rad = start_rad
        self.line_draw_num = line_draw_num
        self.nami1 = nami1
        self.nami2 = nami2
        self.oval_rate_x = oval_rate_x
        self.oval_rate_y = oval_rate_y

    def gen_image(self, idx, profile=False):

        random.seed(idx)
        start_pos_h = (
            self.image_size + random.randint(-1 * self.image_size, self.image_size)
        ) / 2
        start_pos_w = (
            self.image_size + random.randint(-1 * self.image_size, self.image_size)
        ) / 2
        vertex_x = []
        vertex_y = []
        Noise_x = []
        Noise_y = []

        im = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        # Profile time if flag
        if profile:
            t = time.time()

        draw = ImageDraw.Draw(im)
        angle = (math.pi * 2) / self.vertex_number

        for vertex in range(self.vertex_number):
            # Compute param angle
            vertex_x.append(
                math.cos(angle * vertex) * self.start_rad * self.oval_rate_x
                + start_pos_w
            )
            vertex_y.append(
                math.sin(angle * vertex) * self.start_rad * self.oval_rate_y
                + start_pos_h
            )

        vertex_x.append(vertex_x[0])
        vertex_y.append(vertex_y[0])

        for line_draw in range(self.line_draw_num):
            gray = random.randint(0, 255)
            Noise_x.clear()
            Noise_y.clear()
            for vertex in range(self.vertex_number):
                Noise_x.append(random.uniform(0, 10000))
                Noise_x[vertex] = (
                    noise.pnoise1(Noise_x[vertex]) * self.perlin_noise_coefficient
                    - self.perlin_noise_coefficient
                    - 0.5 * math.sin(angle * vertex * self.nami1)
                    + 0.5 * math.sin(angle * vertex * self.nami2)
                )

            for vertex in range(self.vertex_number):
                Noise_y.append(random.uniform(0, 10000))
                Noise_y[vertex] = (
                    noise.pnoise1(Noise_y[vertex]) * self.perlin_noise_coefficient
                    - self.perlin_noise_coefficient
                    - 0.5 * math.sin(angle * vertex * self.nami1)
                    + 0.5 * math.sin(angle * vertex * self.nami2)
                )

            for vertex in range(self.vertex_number):
                vertex_x[vertex] -= math.cos(angle * vertex) * (
                    Noise_x[vertex] - self.line_width
                )
                vertex_y[vertex] -= math.sin(angle * vertex) * (
                    Noise_y[vertex] - self.line_width
                )

            vertex_x[self.vertex_number] = vertex_x[0]
            vertex_y[self.vertex_number] = vertex_y[0]

            for i in range(self.vertex_number):
                draw.line(
                    (vertex_x[i], vertex_y[i], vertex_x[i + 1], vertex_y[i + 1]),
                    fill=(gray, gray, gray),
                    width=1,
                )

        # Profile time if flag
        if profile:
            t = time.time() - t
            return im, t
        else:
            return im


# Create a vectorized version of the pnoise1 function
pnoise1_vectorized = np.vectorize(noise.pnoise1)


class OptimSyntDatasetClass:
    def __init__(
        self,
        image_size,
        vertex_number,
        perlin_noise_coefficient,
        line_width,
        start_rad,
        line_draw_num,
        nami1,
        nami2,
        oval_rate_x,
        oval_rate_y,
    ):

        self.image_size = image_size
        self.vertex_number = vertex_number
        self.perlin_noise_coefficient = perlin_noise_coefficient
        self.line_width = line_width
        self.start_rad = start_rad
        self.line_draw_num = line_draw_num
        self.nami1 = nami1
        self.nami2 = nami2
        self.oval_rate_x = oval_rate_x
        self.oval_rate_y = oval_rate_y

    def gen_image(self, idx, profile=False):

        im = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        np.random.seed(idx)

        start_pos_h = (
            self.image_size + np.random.randint(-self.image_size, self.image_size + 1)
        ) / 2
        start_pos_w = (
            self.image_size + np.random.randint(-self.image_size, self.image_size + 1)
        ) / 2

        # Profile time if flag
        if profile:
            t = time.time()

        draw = ImageDraw.Draw(im)
        angle = (math.pi * 2) / self.vertex_number

        # Vectorized computation
        paramangle = np.arange(self.vertex_number) * angle
        vertex_x = np.cos(paramangle) * self.start_rad * self.oval_rate_x + start_pos_h
        vertex_y = np.sin(paramangle) * self.start_rad * self.oval_rate_y + start_pos_w
        wave = 0.5 * np.sin(paramangle * self.nami1) + 0.5 * np.sin(
            paramangle * self.nami2
        )

        # Pre-allocate a single array for drawing points
        points_to_draw = np.empty((self.vertex_number + 1, 2))
        points_to_draw[: self.vertex_number, 0] = vertex_x
        points_to_draw[: self.vertex_number, 1] = vertex_y
        points_to_draw[self.vertex_number] = points_to_draw[0]

        # New optimization: Pre-generate all gray values in a single call
        gray_values = np.random.randint(0, 256, self.line_draw_num)

        for i in range(self.line_draw_num):
            gray = gray_values[i]

            # Vectorized noise calculation
            noise_x = (
                pnoise1_vectorized(np.random.uniform(0, 10000, self.vertex_number))
                * self.perlin_noise_coefficient
                - self.perlin_noise_coefficient
                - wave
            )
            noise_y = (
                pnoise1_vectorized(np.random.uniform(0, 10000, self.vertex_number))
                * self.perlin_noise_coefficient
                - self.perlin_noise_coefficient
                - wave
            )

            # Vectorized update of vertex positions
            points_to_draw[: self.vertex_number, 0] -= np.cos(paramangle) * (
                noise_x - self.line_width
            )
            points_to_draw[: self.vertex_number, 1] -= np.sin(paramangle) * (
                noise_y - self.line_width
            )
            points_to_draw[self.vertex_number] = points_to_draw[0]

            # Draw the lines
            for j in range(self.vertex_number):
                draw.line(
                    (
                        points_to_draw[j, 0],
                        points_to_draw[j, 1],
                        points_to_draw[j + 1, 0],
                        points_to_draw[j + 1, 1],
                    ),
                    fill=(gray, gray, gray),
                    width=1,
                )

        # Profile time if flag
        if profile:
            t = time.time() - t
            return im, t
        else:
            return im
