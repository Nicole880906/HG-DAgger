import unittest

import numpy as np

from eval_surgflow_offline import error_metrics, select_indices


class SurgFlowOfflineEvalTest(unittest.TestCase):
    def test_select_indices_uniformly_covers_endpoints(self):
        result = select_indices(length=101, stride=1, max_windows=5)
        np.testing.assert_array_equal(result, [0, 25, 50, 75, 100])

    def test_select_indices_applies_stride_before_limit(self):
        result = select_indices(length=10, stride=3, max_windows=None)
        np.testing.assert_array_equal(result, [0, 3, 6, 9])

    def test_error_metrics(self):
        target = np.zeros((2, 2, 2), dtype=np.float32)
        prediction = np.ones_like(target)
        metrics = error_metrics(prediction, target)
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["mae"], 1.0)
        self.assertEqual(metrics["rmse"], 1.0)
        self.assertEqual(metrics["per_dimension_mae"], [1.0, 1.0])
        self.assertEqual(metrics["per_timestep_rmse"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
