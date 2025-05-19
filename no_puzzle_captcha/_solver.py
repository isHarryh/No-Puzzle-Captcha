from cv2.typing import MatLike
from typing import Sequence, Union

import cv2
import os
import time
import numpy as np

from ._transforms import ImageTransform, NormalizeTransform, EdgeTransform

BytesLike = Union[bytes, bytearray, memoryview]
_BytesLikeClasses = (bytes, bytearray, memoryview)


def _show_image(image: MatLike) -> None:
    name = hex(id(image))
    if image.ndim == 2:
        cv2.imshow(f"{name} (Grayscale)", image)
        cv2.waitKey(0)
    elif image.ndim == 3:
        if image.shape[2] == 2:
            cv2.imshow(f"{name} (Ch {0})", image[:, :, 0])
            cv2.imshow(f"{name} (Ch {1})", image[:, :, 1])
            cv2.waitKey(0)
        elif image.shape[2] == 3:
            cv2.imshow(f"{name} (RGB)", image)
            cv2.waitKey(0)
        else:
            raise ValueError("Unsupported channel number")
    else:
        raise ValueError("Unsupported image shape")


class PuzzleCaptchaResult:
    """Holds the result of the captcha solving process."""

    V_THEME_COLOR = tuple(reversed((255, 64, 0)))
    V_TEXT_COLOR = tuple(reversed((255, 255, 255)))
    V_FONT_FAMILY = cv2.FONT_HERSHEY_SIMPLEX
    V_FONT_SCALE = 0.4
    V_FONT_THICKNESS = 1

    def __init__(self, x: int, y: int, background_image: MatLike, puzzle_image: MatLike, elapsed_time: float):
        self._x = x
        self._y = y
        self._background_image = background_image
        self._puzzle_image = puzzle_image
        self._elapsed_time = elapsed_time

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def background_image(self) -> MatLike:
        return self._background_image

    @property
    def puzzle_image(self) -> MatLike:
        return self._puzzle_image

    @property
    def elapsed_time(self) -> float:
        return self._elapsed_time

    def visualize(self) -> MatLike:
        """Draws a red rectangle around the detected area on the background image."""
        img = self.background_image.copy()
        h, w = self.puzzle_image.shape[:2]

        cv2.rectangle(img, (self.x, self.y), (self.x + w, self.y + h), PuzzleCaptchaResult.V_THEME_COLOR, 2)

        label_text = f"({self.x},{self.y})"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            PuzzleCaptchaResult.V_FONT_FAMILY,
            PuzzleCaptchaResult.V_FONT_SCALE,
            PuzzleCaptchaResult.V_FONT_THICKNESS,
        )

        label_rect_tl = (self.x, self.y)
        label_rect_br = (self.x + text_w + 4, self.y + text_h + baseline + 4)
        cv2.rectangle(img, label_rect_tl, label_rect_br, PuzzleCaptchaResult.V_THEME_COLOR, -1)

        text_pos = (self.x + 2, self.y + text_h + baseline - 2)
        cv2.putText(
            img,
            label_text,
            text_pos,
            PuzzleCaptchaResult.V_FONT_FAMILY,
            PuzzleCaptchaResult.V_FONT_SCALE,
            PuzzleCaptchaResult.V_TEXT_COLOR,
        )

        return img

    def visualize_and_save(self, path: str, auto_mkdir: bool = True) -> None:
        """Saves the visualized image to the specified path."""
        img = self.visualize()
        if auto_mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)

    def visualize_and_show(self) -> None:
        """Shows the visualized image."""
        img = self.visualize()
        _show_image(img)


class PuzzleCaptchaSolver:
    """Solves puzzle captchas by applying transformations and template matching."""

    DEFAULT_TRANSFORMS = (NormalizeTransform(), EdgeTransform(150, 250))
    MIN_IMAGE_SIZE = 4
    MAX_IMAGE_SIZE = 8192

    def __init__(self, transforms: Sequence[ImageTransform] = DEFAULT_TRANSFORMS):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms must be a sequence")
        for t in transforms:
            if not isinstance(t, ImageTransform):
                raise TypeError("Each transform must be a ImageTransform instance")
        self.transforms = transforms

    def handle_file(self, background_path: str, puzzle_path: str) -> PuzzleCaptchaResult:
        """Process images from file paths."""
        self._check_file(background_path, puzzle_path)

        background = cv2.imread(background_path)
        puzzle = cv2.imread(puzzle_path)
        return self.handle_image(background, puzzle)

    def handle_bytes(self, background_bytes: BytesLike, puzzle_bytes: BytesLike) -> PuzzleCaptchaResult:
        """Process images from byte data."""
        self._check_bytes(background_bytes, puzzle_bytes)

        background = cv2.imdecode(np.frombuffer(background_bytes, np.uint8), cv2.IMREAD_COLOR)
        puzzle = cv2.imdecode(np.frombuffer(puzzle_bytes, np.uint8), cv2.IMREAD_COLOR)
        return self.handle_image(background, puzzle)

    def handle_image(self, background: MatLike, puzzle: MatLike) -> PuzzleCaptchaResult:
        """Process in-memory images and find the puzzle position."""
        self._check_image(background, puzzle)

        t0 = time.perf_counter()

        processed_background = self._apply_transforms(background)
        processed_puzzle = self._apply_transforms(puzzle)
        result = cv2.matchTemplate(processed_background, processed_puzzle, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc

        return PuzzleCaptchaResult(x, y, background, puzzle, time.perf_counter() - t0)

    def _check_file(self, background_path: str, puzzle_path: str) -> None:
        if not os.path.isfile(background_path):
            raise FileNotFoundError(f"Given background file not found: {background_path}")
        if not os.path.isfile(puzzle_path):
            raise FileNotFoundError(f"Given puzzle file not found: {puzzle_path}")

    def _check_bytes(self, background_bytes: BytesLike, puzzle_bytes: BytesLike) -> None:
        if not isinstance(background_bytes, _BytesLikeClasses):
            raise TypeError(f"Given background bytes must be a bytes-like object, but got {type(background_bytes)}")
        if not isinstance(puzzle_bytes, _BytesLikeClasses):
            raise TypeError(f"Given puzzle bytes must be a bytes-like object, but got {type(puzzle_bytes)}")
        if len(background_bytes) == 0:
            raise ValueError("Given background bytes cannot be empty")
        if len(puzzle_bytes) == 0:
            raise ValueError("Given puzzle bytes cannot be empty")

    def _check_image(self, background: MatLike, puzzle: MatLike) -> None:
        if background is None:
            raise ValueError("Given background image cannot be None")
        if puzzle is None:
            raise ValueError("Given puzzle image cannot be None")

        bh, bw = background.shape[:2]
        ph, pw = puzzle.shape[:2]
        if bw < PuzzleCaptchaSolver.MIN_IMAGE_SIZE or bh < PuzzleCaptchaSolver.MIN_IMAGE_SIZE:
            raise ValueError(f"Given background size ({bw}*{bh}) is too small")
        if bw > PuzzleCaptchaSolver.MAX_IMAGE_SIZE or bh > PuzzleCaptchaSolver.MAX_IMAGE_SIZE:
            raise ValueError(f"Given background size ({bw}*{bh}) is too large")
        if pw < PuzzleCaptchaSolver.MIN_IMAGE_SIZE or ph < PuzzleCaptchaSolver.MIN_IMAGE_SIZE:
            raise ValueError(f"Given puzzle size ({pw}*{ph}) is too small")
        if pw > PuzzleCaptchaSolver.MAX_IMAGE_SIZE or ph > PuzzleCaptchaSolver.MAX_IMAGE_SIZE:
            raise ValueError(f"Given puzzle size ({pw}*{ph}) is too large")
        if pw > bw or ph > bh:
            raise ValueError(f"The puzzle ({pw}*{ph}) is oversized compared to the background ({bw}*{bh})")

    def _apply_transforms(self, image: MatLike) -> MatLike:
        """Applies the sequence of transformations to the image."""
        processed = image.copy()
        for t in self.transforms:
            processed = t.transform(processed)
        return processed
