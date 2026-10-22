import unittest

import numpy as np

from cmu import CACHE, forward, load, numeric, parse_amc, parse_asf, rotation

ASF = """
:version 1.10
:units
mass 1
length 0.45
angle deg
:root
order TX TY TZ RX RY RZ
axis XYZ
position 0 0 0
orientation 0 0 0
:bonedata
begin
id 1
name limb
direction 0 0 1
length 1
axis 0 0 90 XYZ
dof rx
limits (-180 180)
end
:hierarchy
begin
root limb
end
"""
AMC = ":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 0 0 0 0 0\nlimb 90\n2\nroot 0 0 0 0 0 90\nlimb 90\n"


class SafeMocapTests(unittest.TestCase):
    def test_hand_computed_axis_conjugation_and_hierarchy(self):
        skeleton = parse_asf(ASF)
        frames = parse_amc(AMC, skeleton)
        first, rotations = forward(skeleton, frames[0])
        second, _ = forward(skeleton, frames[1])
        scale = 0.0254 / 0.45
        np.testing.assert_allclose(first["limb"], [scale, 0, 0], atol=1e-12)
        np.testing.assert_allclose(second["limb"], [0, scale, 0], atol=1e-12)
        np.testing.assert_allclose(rotations["limb"].T @ rotations["limb"], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(skeleton.scale, 0.05644444444444444)

    def test_rotation_order_is_column_equivalent_of_asf_row_order(self):
        np.testing.assert_allclose(rotation(np.array([90, 90]), ["rx", "ry"]) @ [0, 1, 0],
                                   [1, 0, 0], atol=1e-12)

    def test_reject_non_numeric_or_nonfinite_payloads(self):
        for value in ("NaN", "inf", "1e999", "__import__('os')", "0x42", "[1]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                numeric([value])

    def test_reject_malformed_hierarchy_frames_and_channels(self):
        for text in (ASF.replace("root limb", "limb limb"), ASF.replace("root limb", "root missing"),
                     ASF.replace("angle deg", "angle rad"), ASF.replace("orientation 0 0 0", "orientation 1 0 0")):
            with self.assertRaises(ValueError):
                parse_asf(text)
        skeleton = parse_asf(ASF)
        for text in (AMC.replace("\n2\n", "\n3\n"), AMC.replace("limb 90", "limb 90 10"),
                     AMC.replace("limb 90\n2", "2"), AMC.replace(":DEGREES", ":RADIANS"),
                     AMC.replace("\nlimb 90", "\nunknown 90")):
            with self.assertRaises(ValueError):
                parse_amc(text, skeleton)


@unittest.skipUnless((CACHE / "provenance.json").exists(), "Fetch selected CMU assets for source integration tests.")
class CmuSourceTests(unittest.TestCase):
    def test_pinned_source_scale_duration_and_rigid_skeleton(self):
        skeleton, frames, provenance = load()
        self.assertEqual(len(frames), 449)
        self.assertEqual(provenance["fps"], 120)
        self.assertEqual(len(skeleton.bones), 30)
        roots, ground_points = [], []
        for frame in frames:
            positions, rotations = forward(skeleton, frame)
            roots.append(positions["root"])
            ground_points.extend([positions[name][1] for name in ("lfoot", "rfoot", "ltoes", "rtoes")])
            for name, bone in skeleton.bones.items():
                length = np.linalg.norm(positions[name] - positions[skeleton.parents[name]])
                self.assertAlmostEqual(length, bone.length * skeleton.scale, delta=1e-6)
                self.assertAlmostEqual(np.linalg.det(rotations[name]), 1, delta=1e-12)
        roots = np.asarray(roots)
        self.assertTrue(np.all((roots[:, 1] > 0.85) & (roots[:, 1] < 1.1)))
        self.assertGreater(roots[-1, 2] - roots[0, 2], 3)
        self.assertLess(roots[-1, 2] - roots[0, 2], 4)
        self.assertGreater(min(ground_points), -0.02)
        self.assertLess(min(ground_points), 0.04)


if __name__ == "__main__":
    unittest.main()
