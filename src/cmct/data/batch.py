"""The one batch record every branch receives.

A branch never touches a DataLoader or a dict of tensor keys. Adding a view
means adding a field here, which makes every use of it visible to a type checker
instead of a KeyError at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from torch import Tensor


@dataclass
class Batch:
    img_x: Tensor
    """Labeled source images, weak view. [B, 3, H, W]"""
    label_x: Tensor
    """Source labels. [B]"""
    img_u: Tensor
    """Unlabeled target images, weak view -- the view every teacher-side and
    reference-side computation must use. [B, 3, H, W]"""
    img_u_strong: Tensor | None = None
    """Target images, strong view. Always None for now: strong augmentation is
    not implemented (see StreamConfig.strong_aug)."""

    def to(self, device: str) -> Batch:
        return replace(
            self,
            img_x=self.img_x.to(device),
            label_x=self.label_x.to(device),
            img_u=self.img_u.to(device),
            img_u_strong=None if self.img_u_strong is None else self.img_u_strong.to(device),
        )

    @property
    def student_img_u(self) -> Tensor:
        """`img_u_strong` when present, else `img_u`. The single place that
        fallback is expressed, so no branch reimplements it differently."""
        return self.img_u if self.img_u_strong is None else self.img_u_strong
