"""Check measured-pair selection without interpolating equal-accuracy claims."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from matplotlib.figure import Figure
from warp_fdtd.guide_report import paired_comparisons, main
from warp_fdtd.guide_problem import GuideProblem


class GuideReportTests(unittest.TestCase):
    def test_error_budget_rejects_faster_but_less_accurate_octree(self):
        cases = {
            "u25": dict(
                kind="uniform",
                spacing=0.025,
                median_s=10,
                owned_bytes=1000,
                sampled_field_l2=0.05,
            ),
            "u20": dict(
                kind="uniform",
                spacing=0.02,
                median_s=20,
                owned_bytes=2000,
                sampled_field_l2=0.03,
            ),
            "o25": dict(
                kind="octree",
                spacing=0.025,
                median_s=1,
                owned_bytes=100,
                sampled_field_l2=0.051,
            ),
            "o20": dict(
                kind="octree",
                spacing=0.02,
                median_s=2,
                owned_bytes=200,
                sampled_field_l2=0.031,
            ),
            "reference_only": dict(
                kind="uniform",
                spacing=0.025,
                median_s=None,
                owned_bytes=1000,
                sampled_field_l2=0,
            ),
        }
        fields = {
            k: {"fields": np.full((2, 6, 3), 2.0 if v["kind"] == "uniform" else 3.0)}
            for k, v in cases.items()
        }
        budget = paired_comparisons(cases, fields, np.ones((6, 3)))
        self.assertEqual(budget["u25"]["octree_case"], "o20")
        self.assertEqual(budget["u25"]["speedup"], 5)
        self.assertEqual(budget["u25"]["owned_memory_ratio"], 5)
        self.assertNotIn("u20", budget)
        self.assertNotIn("reference_only", budget)
        self.assertEqual(cases["o25"]["same_finest_uniform_case"], "u25")
        self.assertAlmostEqual(cases["o25"]["same_finest_field_difference"], 0.5)
        self.assertEqual(
            cases["o25"]["same_finest_field_difference_by_time"], [0.5, 0.5]
        )

    def test_second_reference_is_not_treated_as_a_timed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = GuideProblem().parameters()
            for name, spacing, value, runtime in [
                ("u0111", 1 / 90, 1.0, None),
                ("u0125", 0.0125, 0.9, None),
                ("u020", 0.02, 0.8, 2.0),
            ]:
                case = root / name
                case.mkdir()
                (case / "result.json").write_text(
                    json.dumps(
                        dict(
                            kind="uniform",
                            spacing=spacing,
                            problem=problem,
                            median_s=runtime,
                            owned_bytes=1000,
                        )
                    )
                )
                np.savez(
                    case / "fields.npz",
                    fields=np.full((3, 6, 2), value),
                    xyz=np.array([[1.0, 1.6, 1.6], [2.0, 2.1, 1.6]]),
                    times=np.array([6.0, 10.0, 14.0]),
                )
            legends = []
            savefig = Figure.savefig

            def inspect_legend(figure, filename, *args, **kwargs):
                if Path(filename).name == "resolution_trends.png":
                    legends.extend(figure.axes[0].get_legend_handles_labels()[1])
                return savefig(figure, filename, *args, **kwargs)

            with patch("sys.argv", ["guide_report", str(root), "--reference", "u0111"]):
                with patch.object(Figure, "savefig", inspect_legend):
                    main()
            self.assertEqual(legends, ["uniform"])
            check = json.loads((root / "reference_checks.json").read_text())
            self.assertEqual(set(check), {"u0125"})
            self.assertAlmostEqual(check["u0125"]["field_difference"], 0.1)
            report = json.loads((root / "comparison.json").read_text())
            self.assertIsNone(report["u0125"]["median_s"])
            self.assertAlmostEqual(report["u020"]["sampled_field_l2"], 0.2)
            sensitivity = json.loads((root / "reference_sensitivity.json").read_text())
            self.assertAlmostEqual(sensitivity["u0111"]["u020"], 0.2)
            self.assertAlmostEqual(sensitivity["u0125"]["u020"], 1 / 9)
            self.assertTrue((root / "resolution_trends.png").exists())

    def test_no_pair_does_not_invent_a_measurement(self):
        cases = {
            "octree": dict(
                kind="octree",
                spacing=0.02,
                median_s=1,
                owned_bytes=100,
                sampled_field_l2=0.05,
            )
        }
        result = paired_comparisons(cases, {}, np.ones((6, 3)))
        self.assertEqual(result, {})
        self.assertNotIn("same_finest_field_difference", cases["octree"])


if __name__ == "__main__":
    unittest.main()
