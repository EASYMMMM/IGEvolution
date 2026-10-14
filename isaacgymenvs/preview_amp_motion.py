"""Render actual MJCF SMPL/collision geometry offline, without Isaac Gym."""
import argparse
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from motion_tools.amp_motion import MOTIONS, MJCF, load_motion, forward_kinematics


def sphere_triangles(radius, center=(0, 0, 0), rings=8, sectors=12):
    lat = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    lon = np.linspace(0, 2 * np.pi, sectors + 1)
    points = np.stack([np.cos(lat[:, None]) * np.cos(lon),
                       np.cos(lat[:, None]) * np.sin(lon),
                       np.broadcast_to(np.sin(lat[:, None]), (rings + 1, sectors + 1))], axis=-1)
    points = points * radius + center
    return np.array([points[ids] for i in range(rings) for j in range(sectors)
                     for ids in (((i, i + 1, i + 1), (j, j, j + 1)),
                                 ((i, i + 1, i), (j, j + 1, j + 1)))])


def capsule_triangles(radius, a, b):
    a, b = np.asarray(a), np.asarray(b)
    direction = b - a
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return sphere_triangles(radius, a)
    direction /= length
    x = np.cross(direction, [0, 0, 1] if abs(direction[2]) < .9 else [1, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(direction, x)
    angle = np.linspace(0, 2 * np.pi, 13)
    ring = radius * (np.cos(angle[:, None]) * x + np.sin(angle[:, None]) * y)
    cylinder = np.array([triangle for i in range(12)
                         for triangle in ((a + ring[i], b + ring[i], b + ring[i + 1]),
                                          (a + ring[i], b + ring[i + 1], a + ring[i + 1]))])
    return np.concatenate([sphere_triangles(radius, a), sphere_triangles(radius, b), cylinder])


def box_triangles(half):
    p = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([p[list(ids)] for a, b, c, d in faces for ids in ((a, b, c), (a, c, d))])


def read_stl(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].astype(float)


def geometry(xml_path, node_names, smpl=False):
    xml = ET.parse(xml_path).getroot()
    compiler = xml.find("compiler")
    meshdir = xml_path.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    meshes = {m.get("name"): m for m in xml.findall("asset/mesh")}
    result = []
    for j, name in enumerate(node_names):
        for g in xml.find(f".//body[@name='{name}']").findall("geom"):
            is_mesh = g.get("type") == "mesh"
            if is_mesh != smpl:
                continue
            kind = g.get("type", "capsule")
            pos = np.fromstring(g.get("pos", "0 0 0"), sep=" ")
            size = np.fromstring(g.get("size", "0"), sep=" ")
            if kind == "mesh":
                m = meshes[g.get("mesh")]
                triangles = read_stl(meshdir / m.get("file"))
                triangles *= np.fromstring(m.get("scale", "1 1 1"), sep=" ")
            elif kind == "sphere":
                triangles = sphere_triangles(size[0])
            elif kind == "box":
                triangles = box_triangles(size)
            elif kind == "capsule":
                ends = np.fromstring(g.get("fromto", ""), sep=" ")
                if len(ends) == 6:
                    triangles = capsule_triangles(size[0], ends[:3], ends[3:])
                else:
                    triangles = capsule_triangles(size[0], [0, 0, -size[1]], [0, 0, size[1]])
            else:
                raise ValueError(f"Unsupported preview geometry: {kind}")
            wxyz = np.fromstring(g.get("quat", "1 0 0 0"), sep=" ")
            rot = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
            triangles = triangles @ rot.T + pos
            result.append((j, triangles))
    return result


def render_setup(size=(10, 5), dpi=100):
    fig = plt.figure(figsize=size, dpi=dpi, facecolor="#f5f7fa")
    axes, collections = [], []
    for i, title in enumerate(("Original collision model", "Same motion + SMPL meshes")):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_title(title, fontsize=12, pad=0)
        ax.view_init(elev=12, azim=-65)
        ax.set_box_aspect((1, 1, 1.8))
        ax.set_axis_off()
        color = "#6999bd" if i == 0 else "#c5d2e0"
        collection = Poly3DCollection([], facecolors=color, edgecolors="#415366", linewidths=.18, alpha=1)
        ax.add_collection3d(collection)
        axes.append(ax)
        collections.append(collection)
    fig.subplots_adjust(left=0, right=1, bottom=.02, top=.92, wspace=0)
    return fig, axes, collections


def update_frame(fig, axes, collections, geometries, pos, matrices, index, fps):
    center = pos[index, 0].copy()
    center[2] = 0
    for ax, collection, geoms in zip(axes, collections, geometries):
        vertices = np.concatenate([tri @ matrices[index, j].T + pos[index, j] - center for j, tri in geoms])
        # Simple directional lighting preserves the actual triangles and transforms.
        normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        shade = .62 + .38 * np.maximum(0, normal @ np.array([-.3, -.4, .866]))
        base = np.array([.43, .62, .76]) if collection is collections[0] else np.array([.80, .86, .94])
        collection.set_verts(vertices)
        collection.set_facecolor(np.c_[shade[:, None] * base, np.ones(len(shade))])
        ax.set_xlim(-.8, .8)
        ax.set_ylim(-.8, .8)
        ax.set_zlim(-.04, 1.95)
        if not getattr(ax, "_motion_floor", False):
            floor = Poly3DCollection([np.array([[-.8, -.8, 0], [.8, -.8, 0], [.8, .8, 0], [-.8, .8, 0]])],
                                    facecolors="#e1e6ed", edgecolors="#c8d0dc", alpha=.45)
            ax.add_collection3d(floor)
            ax.plot([-.65, -.2], [-.65, -.65], [0, 0], color="#57755b")
            ax.text(-.65, -.7, .02, "+X forward", fontsize=8)
            ax._motion_floor = True
    fig.suptitle(f"{fig._motion_label} | {index / fps:.2f} s | camera follows pelvis | kinematic preview", fontsize=11)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=MOTIONS / "P002_WALK_T01_walk_amp.npy")
    parser.add_argument("--output-dir", type=Path, default=MOTIONS / "previews")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate; AMP data is unchanged")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("Preview FPS must be positive")
    motion = load_motion(args.motion)
    fps = int(motion["fps"])
    pos, matrices = forward_kinematics(motion["rotation"]["arr"], motion["root_translation"]["arr"], motion["skeleton_tree"])
    names = motion["skeleton_tree"]["node_names"]
    geoms = [geometry(MJCF / "amp_humanoid_175_muscle.xml", names),
             geometry(MJCF / "amp_humanoid_175_muscle_smpl.xml", names, smpl=True)]
    start = max(0, int(args.start * fps))
    end = len(pos) if args.duration is None else min(len(pos), start + int(args.duration * fps))
    if end <= start:
        parser.error("Empty preview interval")
    indices = np.unique(np.clip(np.round(np.arange(start, end, fps / args.fps)).astype(int), start, end - 1))
    fig, axes, collections = render_setup()
    fig._motion_label = args.motion.stem
    images = []
    for k, index in enumerate(indices):
        images.append(update_frame(fig, axes, collections, geoms, pos, matrices, index, fps))
        if k % 24 == 0:
            print(f"Rendered {k + 1}/{len(indices)} preview frames", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif = args.output_dir / f"{args.motion.stem}_both.gif"
    images[0].save(gif, save_all=True, append_images=images[1:], duration=round(1000 / args.fps), loop=0)
    poster = args.output_dir / f"{args.motion.stem}_poster.png"
    images[len(images) // 2].save(poster)
    # Choose eight poses through the requested interval, retaining both models.
    samples = [images[i] for i in np.linspace(0, len(images) - 1, 8).astype(int)]
    sheet = Image.new("RGB", (samples[0].width * 2, samples[0].height * 4), "white")
    for i, sample in enumerate(samples):
        sheet.paste(sample, ((i % 2) * sample.width, (i // 2) * sample.height))
    png = args.output_dir / f"{args.motion.stem}_both.png"
    sheet.save(png)
    plt.close(fig)
    print(f"Saved {gif}\nSaved {png}\nSaved {poster}")


if __name__ == "__main__":
    main()
