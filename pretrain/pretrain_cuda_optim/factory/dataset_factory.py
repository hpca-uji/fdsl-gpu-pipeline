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
        batch_size: int,
        nclasses: int,
        device: int,
        gpus: int,
        config: SectionProxy,
    ):
        # Initialize vars
        self.init_datapoints = init_datapoints
        self.batch_size = batch_size
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

        # Create tensors
        vertex_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        norbits_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        ovalx_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )
        ovaly_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )
        freq1_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        freq2_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        noisecoef_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )
        startrad_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        linewidth_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )

        # Create empty list and iterate for nclasses
        classes = []
        for i in range(self.nclasses):

            # Generate random nvertex and norbits
            nvertex = random.randint(self.nvertex_min, self.nvertex_max)
            vertex_tensor[i] = nvertex
            norbits = random.randint(self.norbits_min, self.norbits_max)
            norbits_tensor[i] = norbits

            # Generate oval rates
            ovalx = random.uniform(1, self.oval_max)
            ovalx_tensor[i] = ovalx
            ovaly = random.uniform(1, self.oval_max)
            ovaly_tensor[i] = ovaly

            # Generate sinus frequencies
            freq1 = random.randint(self.freq_min, self.freq_max)
            freq1_tensor[i] = freq1
            while True:
                freq2 = random.randint(self.freq_min, self.freq_max)
                if freq1 != freq2:
                    break
                elif freq1 == 0:
                    break
            freq2_tensor[i] = freq2

            # Generate other parameters
            noisecoef = random.uniform(self.noisecoef_min, self.noisecoef_min + 4)
            noisecoef_tensor[i] = noisecoef
            startrad = random.randint(self.startrad_min, self.startrad_min + 50)
            startrad_tensor[i] = startrad
            line_width = random.uniform(0.0, self.linewidth_max)
            linewidth_tensor[i] = line_width

        classes = [
            vertex_tensor,
            norbits_tensor,
            ovalx_tensor,
            ovaly_tensor,
            freq1_tensor,
            freq2_tensor,
            noisecoef_tensor,
            startrad_tensor,
            linewidth_tensor,
        ]

        return classes

    def generate_label(self, idx) -> torch.Tensor:

        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.nclasses, (self.batch_size,), generator=g)

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
        batch_size: int,
        nclasses: int,
        aug_repeats: int,
        device: int,
        gpus: int,
        config: SectionProxy,
    ):
        # Initialize vars
        self.init_datapoints = init_datapoints
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.aug_repeats = aug_repeats
        self.device = device
        self.gpus = gpus

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

        # Create tensors
        vertex_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        norbits_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        ovalx_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )
        ovaly_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )
        freq1_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        freq2_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        noisecoef_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )
        startrad_tensor = torch.empty(
            [self.nclasses], dtype=torch.int32, device=self.device
        )
        linewidth_tensor = torch.empty(
            [self.nclasses], dtype=torch.float32, device=self.device
        )

        # Create empty list and iterate for nclasses
        classes = []
        for i in range(self.nclasses):

            # Generate random nvertex and norbits
            nvertex = random.randint(self.nvertex_min, self.nvertex_max)
            vertex_tensor[i] = nvertex
            norbits = random.randint(self.norbits_min, self.norbits_max)
            norbits_tensor[i] = norbits

            # Generate oval rates
            ovalx = random.uniform(1, self.oval_max)
            ovalx_tensor[i] = ovalx
            ovaly = random.uniform(1, self.oval_max)
            ovaly_tensor[i] = ovaly

            # Generate sinus frequencies
            freq1 = random.randint(self.freq_min, self.freq_max)
            freq1_tensor[i] = freq1
            while True:
                freq2 = random.randint(self.freq_min, self.freq_max)
                if freq1 != freq2:
                    break
                elif freq1 == 0:
                    break
            freq2_tensor[i] = freq2

            # Generate other parameters
            noisecoef = random.uniform(self.noisecoef_min, self.noisecoef_min + 4)
            noisecoef_tensor[i] = noisecoef
            startrad = random.randint(self.startrad_min, self.startrad_min + 50)
            startrad_tensor[i] = startrad
            line_width = random.uniform(0.0, self.linewidth_max)
            linewidth_tensor[i] = line_width

        classes = [
            vertex_tensor,
            norbits_tensor,
            ovalx_tensor,
            ovaly_tensor,
            freq1_tensor,
            freq2_tensor,
            noisecoef_tensor,
            startrad_tensor,
            linewidth_tensor,
        ]

        return classes

    def generate_label(self, idx) -> torch.Tensor:

        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.nclasses, (self.batch_size,), generator=g)

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
    batch_size: int,
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

    # Create dataset
    if aug_repeats > 1:  # Call repeats iterable datasets
        dataset = VatomDatasetRepeats(
            init_datapoints=init_datapoints,
            batch_size=batch_size,
            nclasses=nclasses,
            aug_repeats=aug_repeats,
            device=device,
            gpus=gpus,
            config=config,
        )
    else:
        dataset = VatomDataset(
            init_datapoints=init_datapoints,
            batch_size=batch_size,
            nclasses=nclasses,
            device=device,
            gpus=gpus,
            config=config,
        )

    return dataset
