import random
import atexit

from configparser import ConfigParser, SectionProxy
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


## Basic dataset for vatoms
class VatomDataset(IterableDataset):
    def __init__(
        self,
        init_datapoints: int,
        nclasses: int,
        device: int,
        gpus: int,
        config: SectionProxy,
    ):
        # Initialize vars
        self.init_datapoints = init_datapoints
        self.nclasses = nclasses
        self.device = device
        self.gpus = gpus

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

        # Generate classes
        self.classes = self.gen_classes()

    def get_prefetch_info(self):
        return self.classes, self.norbits_max, self.nvertex_max

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
                    np.int32(nvertex),
                    np.int32(norbits),
                    np.float32(ovalx),
                    np.float32(ovaly),
                    np.int32(freq1),
                    np.int32(freq2),
                    np.float32(noisecoef),
                    np.int32(startrad),
                    np.float32(line_width),
                ]
            )

        return classes

    def generate_label(self, idx) -> torch.Tensor:

        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.nclasses, (1,), generator=g).item()

    def __iter__(self):

        # Get worker info
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        gpu_id = self.device

        # Compute global workers
        global_worker_id = gpu_id * num_workers + worker_id
        total_workers = num_workers * self.gpus

        # Compute idx for global worker
        idx = self.init_datapoints + global_worker_id

        while True:

            # Compute labels and yield
            class_id = self.generate_label(idx)
            yield idx, class_id

            # Increase idx given total global workers
            idx += total_workers


## Dataset for vatoms with BatchAugmentation with aug repeats
class VatomDatasetRepeats(IterableDataset):
    def __init__(
        self,
        init_datapoints: int,
        nclasses: int,
        aug_repeats: int,
        device: int,
        gpus: int,
        config: SectionProxy,
    ):
        # Initialize vars
        self.init_datapoints = init_datapoints
        self.nclasses = nclasses
        self.aug_repeats = aug_repeats
        self.device = device
        self.gpus = gpus

        # Assert enough GPUs for aug repeats
        assert (
            self.gpus % self.aug_repeats == 0
        ), "aug_repeats must divide number of GPUs exactly (this also enforces GPUs >= aug_repeats)"

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

        # Generate classes
        self.classes = self.gen_classes()

    def get_prefetch_info(self):
        return self.classes, self.norbits_max, self.nvertex_max

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
                    np.int32(nvertex),
                    np.int32(norbits),
                    np.float32(ovalx),
                    np.float32(ovaly),
                    np.int32(freq1),
                    np.int32(freq2),
                    np.float32(noisecoef),
                    np.int32(startrad),
                    np.float32(line_width),
                ]
            )

        return classes

    def generate_label(self, idx) -> torch.Tensor:

        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.nclasses, (1,), generator=g).item()

    def __iter__(self):

        # Compute repeating groups
        groups_per_step = self.gpus // self.aug_repeats

        # Comput GPU-Group mapping
        group_id = self.device // self.aug_repeats

        # Init train step at 0
        step = 0
        while True:

            # Compute idx to return
            idx = self.init_datapoints + step * groups_per_step + group_id

            # Compute labels and yield
            class_id = self.generate_label(idx)
            yield idx, class_id

            # Increase step
            step += 1


# Wrapper function to use from pretrain.py
def create_vatom_dataset(
    dataset_cfg_path: str,
    dataset_cfg_select: str,
    init_datapoints: int,
    nclasses: int,
    aug_repeats: int,
    device: int,
    gpus: int,
) -> IterableDataset:
    """Function that creates a dataset of visual atoms"""

    # Read config file
    configparser = ConfigParser()
    configparser.read(dataset_cfg_path)
    config = configparser[dataset_cfg_select]

    if aug_repeats > 1:  # Call repeat iterable dataset
        dataset = VatomDatasetRepeats(
            init_datapoints=init_datapoints,
            nclasses=nclasses,
            aug_repeats=aug_repeats,
            device=device,
            gpus=gpus,
            config=config,
        )
    else:
        dataset = VatomDataset(
            init_datapoints=init_datapoints,
            nclasses=nclasses,
            device=device,
            gpus=gpus,
            config=config,
        )
    return dataset
