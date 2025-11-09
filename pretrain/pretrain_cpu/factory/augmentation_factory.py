from typing import Any

import torch
from torch.utils.data import default_collate
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

_FILL = 128


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


class DataAugmentationCollateFunction:
    """Class that whose instances work like a function by using __call__.
    This function applies data augmentation using timm RandomErase
    and PyTorch MixUp/Cutmix with custom label_smoothing as a collate_fn of the dataloader
    """

    def __init__(
        self,
        mixup: float,
        cutmix: float,
        label_smoothing: float,
        nclasses: int,
    ):

        # Set up mixup/cutmix
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

        # Set up smoothing attributes
        self.nclasses = nclasses
        self.label_smoothing = label_smoothing

    def __call__(self, batch):

        # Divide batch into inputs and labels
        inputs, labels = default_collate(batch)

        # Apply mix
        if self.mix is not None:

            inputs, labels = self.mix(inputs, labels)

            labels = (
                labels * (1.0 - self.label_smoothing)
                + self.label_smoothing / self.nclasses
            )

        return inputs, labels


def create_data_transform(res, num_ops, magnitude, reprob):

    data_aug = Compose(
        [
            RandomResizedCrop(size=(res, res), interpolation=InterpolationMode.BICUBIC),
            RandAugmentV2WithInvert(num_ops=num_ops, magnitude=magnitude, fill=0),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            RandomErasing(p=reprob, value="random"),
        ]
    )

    return data_aug
