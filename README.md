No-Puzzle-Captcha
==========
Lightweight Library for Slide Puzzle Captcha Solving  
轻量级的滑动拼图验证码自动识别库

![PyPI - Version](https://img.shields.io/pypi/v/no_puzzle_captcha?label=PyPI%20version)
![PyPI - Downloads](https://img.shields.io/pypi/dm/no_puzzle_captcha?label=PyPI%20downloads)

## Introduction

This Python library is designed to solve puzzle CAPTCHAs (aka slide CAPTCHAs), using OpenCV to achieve high inference speed with low computational cost.

### Features

- **Fast** : Average processing time is less than 5ms.
- **Lightweight** : CPU-friendly. No neural network.
- **Minimal-Dependency** : The only direct dependency is `opencv-python`.
- **Well-Designed** : Follows best practices such as type annotations.

## Get Started

### Installation

Install from PyPI:

```pip
pip install no_puzzle_captcha
```

### Minimal Example

Firstly, prepare an background image (the original image) and a puzzle image (the slider image).

You only need 4 lines of code to solve a puzzle CAPTCHA:

```py
from no_puzzle_captcha import PuzzleCaptchaSolver

solver = PuzzleCaptchaSolver()

result = solver.handle_file("background.png", "puzzle.png")

print(f"Matched at ({result.x}, {result.y})")
```

## Usage

### Solver Class

All the following methods in the class `PuzzleCaptchaSolver` returns a `PuzzleCaptchaResult` object:

- *Method* `handle_file`: Accepts two file path parameters (one for background image and another for puzzle image).
- *Method* `handle_bytes`: Accepts two bytes-like parameters that store the image data.
- *Method* `handle_image`: Accepts two matrix-like parameters that store the pixel data of the image.

### Result Class

You can get the recognition result via the class `PuzzleCaptchaResult`:

- *Property* `x` and `y`: The top-left coordinates of the detected result.

You can also visualize the result.

- *Method* `visualize`: Returns a matrix-like object representing the visualized image.
- *Method* `visualize_and_show`: Shows the visualized image via OpenCV's window.
- *Method* `visualize_and_save`: Saves the visualized image to a specified path.

### Advanced Usage

The constructor of the class `PuzzleCaptchaSolver` accepts an optional `transforms` argument, allowing you to customize the image transformation process.

## Benchmark

The following test results are based on version `1.1.1`. The elapsed time may vary depending on the device. You can run `test.py` to reproduce these tests.

### GeeTest Test

This test uses the standard samples from GeeTest, each contains exactly one hollow in the background image.

**Sample:**

| Background Image                      | Puzzle Image                          |
| ------------------------------------- | ------------------------------------- |
| ![](tests/geetest_test/IMG_000_O.png) | ![](tests/geetest_test/IMG_000_P.png) |

**Result:**

| Item                 | Value                             |
| -------------------- | --------------------------------- |
| Test Cases           | 115 items                         |
| Elapsed Time (All)   | 0.004 s/item                      |
| Elapsed Time (Infer) | 0.002 s/item                      |
| Accuracy             | **90.4%** (104 correct, 11 wrong) |

### Tricky Test

This test uses the samples with multiple hollow for misleading purpose, so we call it "tricky".

**Sample:**

| Background Image                     | Puzzle Image                         |
| ------------------------------------ | ------------------------------------ |
| ![](tests/tricky_test/IMG_000_O.png) | ![](tests/tricky_test/IMG_000_P.png) |

**Result:**

| Item                 | Value                           |
| -------------------- | ------------------------------- |
| Test Cases           | 100 items                       |
| Elapsed Time (All)   | 0.003 s/item                    |
| Elapsed Time (Infer) | 0.002 s/item                    |
| Accuracy             | **99.0%** (99 correct, 1 wrong) |

### Tricky Hard Test

This test uses the samples with multiple hollow, but they are more difficult to recognize.

**Sample:**

| Background Image                          | Puzzle Image                              |
| ----------------------------------------- | ----------------------------------------- |
| ![](tests/tricky_hard_test/IMG_000_O.jpg) | ![](tests/tricky_hard_test/IMG_000_P.jpg) |

**Result:**

| Item                 | Value                             |
| -------------------- | --------------------------------- |
| Test Cases           | 190 items                         |
| Elapsed Time (All)   | 0.003 s/item                      |
| Elapsed Time (Infer) | 0.002 s/item                      |
| Accuracy             | **90.5%** (172 correct, 18 wrong) |

## Credits

- [OpenCV](https://opencv.org) - provides a powerful computer vision tool.
- [Vasyl Smutok's Puzzle-Captcha-Solver](https://github.com/vsmutok/Puzzle-Captcha-Solver) - gave me an inspiration.
- [GeeTest](https://www.geetest.com/en/adaptive-captcha-demo) \| [2Captcha](https://2captcha.com/demo/geetest-v4) \| [GoCaptcha](http://gocaptcha.wencodes.com/en/docs/slide-captcha) - provides samples of GeeTest-style puzzle CAPTCHAs.
- [Capy](https://www.capy.me/products/puzzle_captcha) - provides samples of difficult puzzle CAPTCHAS.
- [USTB's API](https://sso.ustb.edu.cn/idp/captcha/getBlockPuzzle) - provides samples of tricky puzzle CAPTCHAs.
- [Chaoxing's API](https://captcha.chaoxing.com/captcha/get/verification/image) - provides samples of tricky-hard puzzle CAPTCHAs.

## Licensing

This project is licensed under the MIT License. See the [License](https://github.com/isHarryh/No-Puzzle-Captcha/blob/main/LICENSE) file for more details.
