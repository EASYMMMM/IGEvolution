import itertools
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from isaacgymenvs.srl_mjcf_generator.generator.real_hri_v2 import (
    DEFAULT_ASSET, generate_real_hri_v2_model, read_real_hri_v2_design,
)


class RealHRIV2MorphologyTests(unittest.TestCase):
    def generate(self, params, source=DEFAULT_ASSET):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        filename = generate_real_hri_v2_model(params, 'candidate', directory.name, source)
        return Path(filename), ET.parse(filename).getroot()

    def test_both_baselines_preserve_asset_and_resolve_all_meshes(self):
        short_mount = DEFAULT_ASSET.with_name('srl_real_hri_v2_HXYK_175_03dis_mesh.xml')
        for source, distance in ((DEFAULT_ASSET, 0.6), (short_mount, 0.3)):
            with self.subTest(source=source):
                params = read_real_hri_v2_design(source)
                self.assertEqual(params['base_distance'], distance)
                filename, candidate = self.generate(params, source)
                original = ET.parse(source).getroot()
                original_dir = source.parent / original.find('compiler').get('meshdir')
                for mesh in original.findall('asset/mesh'):
                    mesh.set('file', str((original_dir / mesh.get('file')).resolve(strict=True)))
                for mesh in candidate.findall('asset/mesh'):
                    mesh.set('file', str((filename.parent / mesh.get('file')).resolve(strict=True)))
                original.find('compiler').set('meshdir', '.')
                self.assertEqual(ET.tostring(candidate), ET.tostring(original))

    def test_leg_geometry_uses_v2_frames_and_toe_clearance(self):
        params = dict(read_real_hri_v2_design(), leg1_length=0.43, leg2_length=0.51,
                      base_distance=0.4, base_width=0.12)
        filename, candidate = self.generate(params)
        measured = read_real_hri_v2_design(filename)
        for name, value in params.items():
            self.assertAlmostEqual(measured[name], value)
        mount = candidate.find(".//body[@name='SRL_root']")
        for hinge in mount.findall('joint'):
            self.assertEqual(hinge.get('pos'), '0.4 0 0')
        for side, sign in (('left', 1), ('right', -1)):
            hip = mount.find(f"body[@name='{side}_hip_x_link']")
            upper = hip.find(f"body[@name='{side}_hip_y_link']")
            lower = upper.find(f"body[@name='{side}_knee_link']")
            foot = lower.find(f"body[@name='SRL_{side}_end']")
            self.assertAlmostEqual(float(hip.get('pos').split()[1]), sign * 0.2195)
            self.assertAlmostEqual(float(lower.get('pos').split()[2]), -0.43)
            self.assertAlmostEqual(float(foot.get('pos').split()[0]), 0.52)
            for name, axis, ratio in ((upper.get('name'), 2, 0.43 / 0.671),
                                      (lower.get('name'), 0, 0.51 / 0.64)):
                mesh = candidate.find(f"asset/mesh[@name='{name}']")
                self.assertAlmostEqual(float(mesh.get('scale').split()[axis]), ratio)

    def test_all_hinge_combinations_keep_actuated_joint_order(self):
        source = ET.parse(DEFAULT_ASSET).getroot()
        expected = [j.get('name') for j in source.findall('worldbody//joint')]
        for switches in itertools.product((0, 1), repeat=3):
            params = read_real_hri_v2_design()
            for axis, value in zip('zyx', switches):
                params[f'enable_freejoint_{axis}'] = value
            _, candidate = self.generate(params)
            names = [j.get('name') for j in candidate.findall('worldbody//joint')]
            self.assertEqual(names[:31], expected[:31])  # Human DOFs and three load-cell slides.
            self.assertEqual(names[31:-6], [f'SRL_freejoint_{a}' for a, on in zip('zyx', switches) if on])
            self.assertEqual(names[-6:], expected[-6:])


if __name__ == '__main__':
    unittest.main()
