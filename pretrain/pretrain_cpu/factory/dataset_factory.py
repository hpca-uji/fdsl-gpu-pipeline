import random
import math

from configparser import ConfigParser, SectionProxy
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from PIL import Image, ImageDraw
import noise

# from numba import njit, prange, types


def genvatom(
    img,
    nvertex,
    norbits,
    ovalx,
    ovaly,
    freq1,
    freq2,
    noisecoef,
    startrad,
    line_width,
    res,
    idx,
):

    random.seed(idx)
    start_pos_h = (res + random.randint(-1 * res, res)) / 2
    start_pos_w = (res + random.randint(-1 * res, res)) / 2
    angle_array = []
    vertex_x = []
    vertex_y = []
    wave = []
    Noise_x = []
    Noise_y = []

    draw = ImageDraw.Draw(img)
    angle = (math.pi * 2) / nvertex

    for vertex in range(nvertex):
        # Compute param angle
        paramangle = angle * vertex
        vertex_x.append(math.cos(paramangle) * startrad * ovalx + start_pos_h)
        vertex_y.append(math.sin(paramangle) * startrad * ovaly + start_pos_w)
        wave.append(
            0.5 * math.sin(paramangle * freq1) + 0.5 * math.sin(paramangle * freq2)
        )
        angle_array.append(paramangle)

    vertex_x.append(vertex_x[0])
    vertex_y.append(vertex_y[0])

    for line_draw in range(norbits):
        gray = random.randint(0, 255)
        Noise_x.clear()
        Noise_y.clear()
        for vertex in range(nvertex):
            Noise_x.append(random.uniform(0, 10000))
            Noise_x[vertex] = (
                noise.pnoise1(Noise_x[vertex]) * noisecoef - noisecoef - wave[vertex]
            )

        for vertex in range(nvertex):
            Noise_y.append(random.uniform(0, 10000))
            Noise_y[vertex] = (
                noise.pnoise1(Noise_y[vertex]) * noisecoef - noisecoef - wave[vertex]
            )

        for vertex in range(nvertex):
            vertex_x[vertex] -= math.cos(angle_array[vertex]) * (
                Noise_x[vertex] - line_width
            )
            vertex_y[vertex] -= math.sin(angle_array[vertex]) * (
                Noise_y[vertex] - line_width
            )

        vertex_x[nvertex] = vertex_x[0]
        vertex_y[nvertex] = vertex_y[0]

        for i in range(nvertex):
            draw.line(
                (vertex_x[i], vertex_y[i], vertex_x[i + 1], vertex_y[i + 1]),
                fill=(gray, gray, gray),
                width=1,
            )


# Create a vectorized version of the pnoise1 function
pnoise1_vectorized = np.vectorize(noise.pnoise1)


def genvatom_optimized(
    img,
    nvertex,
    norbits,
    ovalx,
    ovaly,
    freq1,
    freq2,
    noisecoef,
    startrad,
    line_width,
    res,
    idx,
):
    np.random.seed(idx)

    start_pos_h = (res + np.random.randint(-res, res + 1)) / 2
    start_pos_w = (res + np.random.randint(-res, res + 1)) / 2

    draw = ImageDraw.Draw(img)
    angle = (math.pi * 2) / nvertex

    # Vectorized computation
    paramangle = np.arange(nvertex) * angle
    vertex_x = np.cos(paramangle) * startrad * ovalx + start_pos_h
    vertex_y = np.sin(paramangle) * startrad * ovaly + start_pos_w
    wave = 0.5 * np.sin(paramangle * freq1) + 0.5 * np.sin(paramangle * freq2)

    # Pre-allocate a single array for drawing points
    points_to_draw = np.empty((nvertex + 1, 2))
    points_to_draw[:nvertex, 0] = vertex_x
    points_to_draw[:nvertex, 1] = vertex_y
    points_to_draw[nvertex] = points_to_draw[0]

    # New optimization: Pre-generate all gray values in a single call
    gray_values = np.random.randint(0, 256, norbits)

    for i in range(norbits):
        gray = gray_values[i]

        # Vectorized noise calculation
        noise_x = (
            pnoise1_vectorized(np.random.uniform(0, 10000, nvertex)) * noisecoef
            - noisecoef
            - wave
        )
        noise_y = (
            pnoise1_vectorized(np.random.uniform(0, 10000, nvertex)) * noisecoef
            - noisecoef
            - wave
        )

        # Vectorized update of vertex positions
        points_to_draw[:nvertex, 0] -= np.cos(paramangle) * (noise_x - line_width)
        points_to_draw[:nvertex, 1] -= np.sin(paramangle) * (noise_y - line_width)
        points_to_draw[nvertex] = points_to_draw[0]

        # Draw the lines
        for j in range(nvertex):
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


## Basic dataset for vatoms
class VatomDataset(IterableDataset):
    def __init__(
        self,
        init_datapoints: int,
        nclasses: int,
        res: int,
        device: int,
        gpus: int,
        config: SectionProxy,
        aug_repeats: int,
    ):
        # Initialize vars
        self.init_datapoints = init_datapoints
        self.nclasses = nclasses
        self.data_aug = None
        self.res = res
        self.device = device
        self.gpus = gpus
        self.aug_repeats = aug_repeats

        # Get vars from config
        self.nvertex_min = config.getint("nvertex_min")
        self.nvertex_max = config.getint("nvertex_max")
        self.norbits_min = config.getint("norbits_min")
        self.norbits_max = config.getint("norbits_max")
        self.oval_max = config.getint("oval_max")
        self.freq_min = config.getint("freq_min")
        self.freq_max = config.getint("freq_max")
        self.noisecoef_min = config.getint("noisecoef_min")
        self.startrad_min = config.getint("startrad_min")
        self.linewidth_max = config.getfloat("linewidth_max")
        self.seed = config.getint("seed")

        # Compute maximum number of vertex per vatom
        self.totalvertex_max = self.nvertex_max * self.norbits_max

        # Generate classes
        self.classes = self.gen_classes()

    def generate_image(self, idx, class_id):

        # Create empty image
        img = Image.new("RGB", (self.res, self.res), (0, 0, 0))

        genvatom_optimized(
            img,
            *self.classes[class_id],
            self.res,
            idx,
        )

        return img

    def generate_label(self, idx) -> torch.Tensor:

        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.nclasses, (1,), generator=g).item()

    def gen_classes(self) -> list[list]:
        """Function that creates array with different class parameters"""

        # Set generation seed
        random.seed(self.seed)

        # Create empty list and iterate for nclasses
        classes = []
        for i in range(self.nclasses):

            # Generate random nvertex and norbits
            nvertex = random.randint(self.nvertex_min, self.nvertex_max)
            norbits = random.randint(self.norbits_min, self.norbits_max)

            # Generate oval rates
            ovalx = random.uniform(1, self.oval_max)
            ovaly = random.uniform(1, self.oval_max)

            # Generate sinus frequencies
            freq1 = random.randint(self.freq_min, self.freq_max)
            while True:
                freq2 = random.randint(self.freq_min, self.freq_max)
                if freq1 != freq2:
                    break
                elif freq1 == 0:
                    break

            # Generate other parameters
            noisecoef = random.uniform(self.noisecoef_min, self.noisecoef_min + 4)
            startrad = random.randint(self.startrad_min, self.startrad_min + 50)
            line_width = random.uniform(0.0, self.linewidth_max)

            # Add class to list
            classes.append(
                [
                    nvertex,
                    norbits,
                    ovalx,
                    ovaly,
                    freq1,
                    freq2,
                    noisecoef,
                    startrad,
                    line_width,
                ]
            )

        return classes

    def __iter__(self):

        worker_info = get_worker_info()
        workers = worker_info.num_workers
        worker_id = worker_info.id

        # Global sample index to start from
        idx = self.init_datapoints + worker_id * self.gpus

        while True:

            # Compute label
            class_id = self.generate_label(idx + self.device)

            # Gen image
            img = self.generate_image(idx + self.device, class_id)
            # Apply data_aug if needed
            if self.data_aug:
                img = self.data_aug(img)

            yield img, class_id

            idx += workers * self.gpus


class BAVatomDataset(VatomDataset):

    def __iter__(self):

        # Get worker info
        worker_info = get_worker_info()
        workers = worker_info.num_workers
        worker_id = worker_info.id

        # Global sample index to start from
        idx = self.init_datapoints + worker_id * self.gpus

        while True:

            shift = self.device
            for i in range(self.aug_repeats):

                # Compute label
                class_id = self.generate_label(idx + shift)

                # Gen image
                img = self.generate_image(idx + shift, class_id)

                # Apply data_aug if needed
                if self.data_aug:
                    img = self.data_aug(img)

                yield img, class_id

                # Increase shift use modulus
                shift = (shift + 1) % self.gpus

            # Increase counter for next iteration
            idx += workers * self.gpus


# Wrapper function to use from pretrain.py
def create_vatom_dataset(
    dataset_cfg_path: str,
    dataset_cfg_select: str,
    init_datapoints: int,
    nclasses: int,
    res: int,
    device: int,
    gpus: int,
    aug_repeats: int,
) -> IterableDataset:
    """Function that creates a dataset of visual atoms"""

    # Read config file
    configparser = ConfigParser()
    configparser.read(dataset_cfg_path)
    config = configparser[dataset_cfg_select]

    # Create dataset
    if aug_repeats >= 2:

        # Check parameters
        assert gpus >= aug_repeats, "One GPU needed for each different augmentation"

        dataset = BAVatomDataset(
            init_datapoints=init_datapoints,
            nclasses=nclasses,
            res=res,
            device=device,
            gpus=gpus,
            config=config,
            aug_repeats=aug_repeats,
        )
    else:
        dataset = VatomDataset(
            init_datapoints=init_datapoints,
            nclasses=nclasses,
            res=res,
            device=device,
            gpus=gpus,
            config=config,
            aug_repeats=aug_repeats,
        )

    return dataset
