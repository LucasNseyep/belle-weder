import torch


class WrapTranslation:
    """
    Circular shift by a random fraction of the image size.

    Picklable (unlike T.Lambda) so it works safely with num_workers > 0.
    """

    def __init__(self, max_frac: float):
        self.max_frac = max_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        shift_h = int(torch.randint(
            -int(self.max_frac * h), int(self.max_frac * h) + 1, (1,)
        ).item())
        shift_w = int(torch.randint(
            -int(self.max_frac * w), int(self.max_frac * w) + 1, (1,)
        ).item())
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(-2, -1))
