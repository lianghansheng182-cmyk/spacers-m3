import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluate_mini_eval import parse_grounding_bbox


class GroundingBBoxParserTest(unittest.TestCase):
    def assert_bbox_almost_equal(self, actual, expected, places=4):
        self.assertIsNotNone(actual)
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_grounding_003_bbox_2d_output(self):
        result = parse_grounding_bbox(
            """```json
[
  {"bbox_2d": [869, 298, 924, 363], "label": "vehicle"}
]
```"""
        )

        self.assertEqual(result.raw_parsed_bbox, [869.0, 298.0, 924.0, 363.0])
        self.assertEqual(result.detected_bbox_space, "pixel_0_1000")
        self.assertEqual(result.parse_status, "ok")
        self.assert_bbox_almost_equal(
            result.normalized_bbox_0_100,
            [86.9, 29.8, 92.4, 36.3],
        )

    def test_grounding_006_bbox_2d_output(self):
        result = parse_grounding_bbox(
            """```json
[
  {"bbox_2d": [629, 297, 685, 364], "label": "vehicle"}
]
```"""
        )

        self.assertEqual(result.raw_parsed_bbox, [629.0, 297.0, 685.0, 364.0])
        self.assertEqual(result.detected_bbox_space, "pixel_0_1000")
        self.assert_bbox_almost_equal(
            result.normalized_bbox_0_100,
            [62.9, 29.7, 68.5, 36.4],
        )

    def test_grounding_008_bbox_3d_output(self):
        result = parse_grounding_bbox('```json\n{ "bbox_3d": [264, 48, 353, 99] }\n```')

        self.assertEqual(result.raw_parsed_bbox, [264.0, 48.0, 353.0, 99.0])
        self.assertEqual(result.detected_bbox_space, "pixel_0_1000")
        self.assert_bbox_almost_equal(
            result.normalized_bbox_0_100,
            [26.4, 4.8, 35.3, 9.9],
        )

    def test_box_1_field_does_not_leak_field_number(self):
        result = parse_grounding_bbox(
            """```json
{
  "box_1": {
    "x1": 256,
    "y1": 424,
    "x2": 336,
    "y2": 584
  }
}
```"""
        )

        self.assertEqual(result.raw_parsed_bbox, [256.0, 424.0, 336.0, 584.0])
        self.assertNotEqual(result.raw_parsed_bbox[0], 1.0)
        self.assert_bbox_almost_equal(
            result.normalized_bbox_0_100,
            [25.6, 42.4, 33.6, 58.4],
        )

    def test_nested_array_under_object_label(self):
        result = parse_grounding_bbox('```json\n{"vehicle": [[12, 395, 103, 607]]}\n```')

        self.assertEqual(result.raw_parsed_bbox, [12.0, 395.0, 103.0, 607.0])
        self.assert_bbox_almost_equal(
            result.normalized_bbox_0_100,
            [1.2, 39.5, 10.3, 60.7],
        )

    def test_invalid_output_fails(self):
        result = parse_grounding_bbox("I cannot find the vehicle.")

        self.assertIsNone(result.raw_parsed_bbox)
        self.assertIsNone(result.normalized_bbox_0_100)
        self.assertEqual(result.parse_status, "failed")
        self.assertIsNotNone(result.parse_error)

    def test_coordinate_scale_0_to_1(self):
        result = parse_grounding_bbox('{"bbox": [0.1, 0.2, 0.3, 0.4]}')

        self.assertEqual(result.detected_bbox_space, "normalized_0_1")
        self.assert_bbox_almost_equal(result.normalized_bbox_0_100, [10, 20, 30, 40])

    def test_coordinate_scale_0_to_100(self):
        result = parse_grounding_bbox('{"bbox": [10, 20, 30, 40]}')

        self.assertEqual(result.detected_bbox_space, "normalized_0_100")
        self.assert_bbox_almost_equal(result.normalized_bbox_0_100, [10, 20, 30, 40])

    def test_coordinate_scale_0_to_1000(self):
        result = parse_grounding_bbox('{"box": "100,200,300,400"}')

        self.assertEqual(result.detected_bbox_space, "pixel_0_1000")
        self.assert_bbox_almost_equal(result.normalized_bbox_0_100, [10, 20, 30, 40])

    def test_wrong_coordinate_order_is_reordered(self):
        result = parse_grounding_bbox('{"bbox": [40, 60, 10, 20]}')

        self.assertEqual(result.parse_status, "ok_reordered")
        self.assert_bbox_almost_equal(result.normalized_bbox_0_100, [10, 20, 40, 60])

    def test_out_of_range_coordinates_fail_without_clipping(self):
        result = parse_grounding_bbox('{"bbox": [0, 20, 1200, 400]}')

        self.assertEqual(result.parse_status, "failed")
        self.assertIsNone(result.normalized_bbox_0_100)
        self.assertIn("exceed", result.parse_error)


if __name__ == "__main__":
    unittest.main()
