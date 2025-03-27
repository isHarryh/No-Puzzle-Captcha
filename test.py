import json
import math
import os
import time

from no_puzzle_captcha import PuzzleCaptchaSolver

for root_path in ["tests/geetest_test", "tests/tricky_test"]:
    dataset = os.path.join(root_path, "dataset.json")

    test: dict = json.load(open(dataset))

    test_name = test["name"]
    test_error_tolerance = test["error_tolerance"]
    test_cases = test["cases"]

    print(f"Processing {test_name}...\n")

    cnt_failed = 0
    cnt_success = 0
    total_elapsed_time = 0
    total_success_bias = [0, 0]

    solver = PuzzleCaptchaSolver()

    start_time = time.perf_counter()
    for case in test_cases:
        bg_img = os.path.normpath(os.path.join(root_path, case["background"]))
        puzzle_img = os.path.normpath(os.path.join(root_path, case["puzzle"]))
        std_x, std_y = case["position"]
        rst = solver.handle_file(bg_img, puzzle_img)
        total_elapsed_time += rst.elapsed_time
        error = math.hypot(std_x - rst.x, std_y - rst.y)
        if error <= test_error_tolerance:
            cnt_success += 1
            total_success_bias[0] += std_x - rst.x
            total_success_bias[1] += std_y - rst.y
        else:
            print(
                f"- Wrong Answer" +
                f"\torigin={bg_img}" +
                f"\tstd=({std_x}, {std_y})" +
                f"\tinfer=({rst.x}, {rst.y})" +
                f"\terror={error:.1f}"
            )
            cnt_failed += 1
            # rst.visualize_and_show()
    total_time = time.perf_counter() - start_time

    cnt_processed = cnt_failed + cnt_success
    accuracy = cnt_success / cnt_processed

    avg_bias_x = total_success_bias[0] / cnt_success if cnt_success else 0.0
    avg_bias_y = total_success_bias[1] / cnt_success if cnt_success else 0.0

    print(f"\nProcessed {cnt_processed} test cases\n")
    print(
        f"Elapsed Time (All)  " +
        f"\t{total_time:.3f}s" +
        f"\t({total_time / cnt_processed:.3f} s/i, {cnt_processed / total_time:.1f} i/s)"
    )
    print(
        f"Elapsed Time (Infer)" +
        f"\t{total_elapsed_time:.3f}s" +
        f"\t({total_elapsed_time / cnt_processed:.3f} s/i, {cnt_processed / total_elapsed_time:.1f} i/s)"
    )
    print(
        f"Accuracy            " +
        f"\t{accuracy:.1%}\t({cnt_success} correct, {cnt_failed} wrong)"
    )
    print(
        f"Label Bias          " +
        f"\t{math.hypot(avg_bias_x, avg_bias_y):.3f}" +
        f"\t(x: {avg_bias_x:.3f}, y: {avg_bias_y:.3f})" +
        "\n"
    )
