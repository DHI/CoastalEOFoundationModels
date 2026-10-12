"""`ExtendedVisionDataset` and its decoders from dinov3 (FINO branch, commit `81aae9a`).

Copied from `dinov3/data/datasets/{extended,decoders}.py`; `bundle_dataset` uses it when
the dinov3 clone is not importable. DINOv3 License Agreement, (c) Meta Platforms, Inc.
"""

from io import BytesIO
from typing import Any

from PIL import Image
from torchvision.datasets import VisionDataset


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image.Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any) -> None:
        self._target = target

    def decode(self) -> Any:
        return self._target


class ExtendedVisionDataset(VisionDataset):
    def __init__(
        self,
        image_decoder: type[Decoder] = ImageDataDecoder,
        target_decoder: type[Decoder] = TargetDecoder,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.image_decoder = image_decoder
        self.target_decoder = target_decoder

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self.image_decoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
