import cv2

from abc import ABC, abstractmethod
from cv2.typing import MatLike


class ImageTransform(ABC):
    """Base class for image transformations."""

    @abstractmethod
    def transform(self, image: MatLike) -> MatLike:
        """Transform the input image and return the processed image."""
        raise NotImplementedError()


class RawTransform(ImageTransform):
    """Returns the input image unchanged."""

    def transform(self, image: MatLike) -> MatLike:
        return image.copy()


class NormalizeTransform(ImageTransform):
    """Applies min-max normalization to a image."""

    def transform(self, image: MatLike) -> MatLike:
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


class EdgeTransform(ImageTransform):
    """Applies Canny edge detection to a image, returning a 1-channel edge map."""

    def __init__(self, low_threshold: int, high_threshold: int):
        self._low = low_threshold
        self._high = high_threshold

    def transform(self, image: MatLike) -> MatLike:
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, self._low, self._high)
        return edges
