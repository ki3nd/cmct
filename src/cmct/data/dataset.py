"""list[Datum] + transforms -> tensors."""
from __future__ import annotations

from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from cmct.data.datum import Datum


def read_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class ImageListDataset(Dataset):
    """Applies every transform to the SAME decoded image.

    Given n transforms, __getitem__ returns keys "img", "img2", ... "imgN" --
    one disk read and one decode per sample regardless of n. That single-decode
    property is what makes a multi-view pair meaningful: the views differ only by
    augmentation, never by which file was read.
    """

    def __init__(self, items: list[Datum], transforms: Any) -> None:
        self.items = items
        self.transforms = transforms if isinstance(transforms, (list, tuple)) else [transforms]
        if not self.transforms:
            raise ValueError("ImageListDataset needs at least one transform")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = read_image(item.path)
        output: dict[str, Any] = {"label": item.label, "index": index}
        for i, tfm in enumerate(self.transforms):
            output["img" if i == 0 else f"img{i + 1}"] = tfm(image)
        return output
