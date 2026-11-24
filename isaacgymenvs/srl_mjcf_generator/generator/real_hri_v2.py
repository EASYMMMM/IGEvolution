"""Generate morphology candidates from the same Real HRI V2 asset used for training."""
import math
import os
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET


_ASSETS = Path(__file__).resolve().parents[3] / 'assets'
DEFAULT_ASSET = _ASSETS / 'mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml'
DESIGN_PARAM_NAMES = (
    'leg1_length', 'leg2_length', 'base_width', 'base_distance',
    'enable_freejoint_z', 'enable_freejoint_y', 'enable_freejoint_x',
)
# The nominal 95 mm base-width parameter places the V2 hip pivots at +/-194.5 mm.
_HIP_MOUNT_OFFSET = 0.0995


def _read_asset(source_asset):
    path = Path(source_asset or DEFAULT_ASSET).resolve()
    root = ET.parse(path).getroot()
    if not root.get('model', '').startswith('srl_real_hri_v2'):
        raise ValueError(f'Expected a Real HRI V2 source asset, got {path}')
    return path, root


def _numbers(element, attribute):
    return [float(word) for word in element.attrib[attribute].split()]


def _set_component(element, attribute, index, value):
    words = element.attrib[attribute].split()
    if not math.isclose(float(words[index]), value, rel_tol=1e-12, abs_tol=1e-12):
        words[index] = format(value, '.12g')
        element.set(attribute, ' '.join(words))


def _design(root):
    mount = root.find(".//body[@name='SRL_root']")
    upper = mount.find(".//body[@name='left_hip_y_link']/geom[@class='collision']")
    lower = mount.find(".//body[@name='left_knee_link']/geom[@class='collision']")
    hip = mount.find("body[@name='left_hip_x_link']")
    return dict(
        leg1_length=-_numbers(upper, 'fromto')[5],
        leg2_length=_numbers(lower, 'fromto')[3],
        base_width=_numbers(hip, 'pos')[1] - _HIP_MOUNT_OFFSET,
        base_distance=-_numbers(mount, 'pos')[0],
        **{f'enable_freejoint_{a}': int(mount.find(f"joint[@name='SRL_freejoint_{a}']") is not None)
           for a in 'zyx'},
    )


def read_real_hri_v2_design(source_asset=None):
    """Measure the configured reference model, including its mounting distance."""
    _, root = _read_asset(source_asset)
    return _design(root)


def _set_attachment(mount, params):
    distance = params['base_distance']
    _set_component(mount, 'pos', 0, -distance)
    # New passive axes use the V2 hinge dynamics and the same pelvis-frame pivot.
    prototype = mount.find('joint')
    if prototype is None:
        prototype = ET.parse(DEFAULT_ASSET).find(".//joint[@name='SRL_freejoint_z']")
    for axis in 'zyx':
        name = f'SRL_freejoint_{axis}'
        joint = mount.find(f"joint[@name='{name}']")
        if joint is not None:
            mount.remove(joint)
        if params[f'enable_freejoint_{axis}']:
            if joint is None:
                joint = deepcopy(prototype)
                joint.set('name', name)
                joint.set('axis', ' '.join('1' if a == axis else '0' for a in 'xyz'))
            _set_component(joint, 'pos', 0, distance)
            mount.insert(len(mount.findall('joint')), joint)


def _resize_link(root, body, child, axis, length_ratio):
    # The historical search holds CAD link masses and inertias fixed and varies
    # the joint spacing. Apply the same length change to contacts and visuals.
    if math.isclose(length_ratio, 1.0, rel_tol=1e-12):
        return
    collision = body.find("geom[@class='collision']")
    end = _numbers(collision, 'fromto')[3 + axis]
    offset = _numbers(child, 'pos')[axis]
    _set_component(collision, 'fromto', 3 + axis, end * length_ratio)
    # Preserve the fixed toe clearance past the collision capsule.
    _set_component(child, 'pos', axis, offset + end * (length_ratio - 1))
    mesh_name = body.find("geom[@class='visual']").attrib['mesh']
    mesh = root.find(f"asset/mesh[@name='{mesh_name}']")
    scale = mesh.get('scale', '1 1 1').split()
    scale[axis] = format(float(scale[axis]) * length_ratio, '.12g')
    mesh.set('scale', ' '.join(scale))


def generate_real_hri_v2_model(srl_params, output_name, output_dir, source_asset=None):
    """Write a candidate MJCF, preserving V2 human, HMI sensors and motor order."""
    source, root = _read_asset(source_asset)
    baseline = _design(root)
    params = {name: float(srl_params[name]) for name in DESIGN_PARAM_NAMES}
    for name in DESIGN_PARAM_NAMES[:4]:
        if not math.isfinite(params[name]) or params[name] <= 0:
            raise ValueError(f'{name} must be positive and finite, got {params[name]}')
    for name in DESIGN_PARAM_NAMES[4:]:
        if params[name] not in (0, 1):
            raise ValueError(f'{name} must be 0 or 1, got {params[name]}')

    dest = Path(output_dir, output_name)
    if dest.suffix != '.xml':
        dest = dest.with_name(dest.name + '.xml')
    dest = dest.resolve()
    if dest == source:
        raise ValueError(f'Candidate output would overwrite its source asset: {source}')
    mount = root.find(".//body[@name='SRL_root']")
    _set_attachment(mount, params)
    for side, sign in (('left', 1), ('right', -1)):
        hip = mount.find(f"body[@name='{side}_hip_x_link']")
        _set_component(hip, 'pos', 1, sign * (params['base_width'] + _HIP_MOUNT_OFFSET))
        upper = hip.find(f"body[@name='{side}_hip_y_link']")
        lower = upper.find(f"body[@name='{side}_knee_link']")
        foot = lower.find(f"body[@name='SRL_{side}_end']")
        _resize_link(root, upper, lower, 2, params['leg1_length'] / baseline['leg1_length'])
        _resize_link(root, lower, foot, 0, params['leg2_length'] / baseline['leg2_length'])

    # Both Isaac Gym and MJCF readers can resolve each file from the output XML.
    compiler = root.find('compiler')
    mesh_dir = source.parent / compiler.get('meshdir', '.')
    for mesh in root.findall('asset/mesh'):
        path = (mesh_dir / mesh.attrib['file']).resolve(strict=True)
        mesh.set('file', os.path.relpath(path, dest.parent))
    compiler.set('meshdir', '.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(dest, encoding='utf-8')
    return str(dest)
