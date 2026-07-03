import argparse
from pathlib import Path

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state


def _default_xml_candidates():
    script_dir = Path(__file__).resolve().parent
    return [
        script_dir.parents[4] / "mjcf" / "amp_humanoid_175_v2.xml",
        script_dir.parents[3] / "assets" / "mjcf" / "amp_humanoid_175_v2.xml",
    ]


def _resolve_default_xml():
    for candidate in _default_xml_candidates():
        if candidate.exists():
            return candidate
    return _default_xml_candidates()[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a SkeletonState T-pose for amp_humanoid_175_v2.xml."
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=str(_resolve_default_xml()),
        help="Path to the target v2 MJCF XML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/amp_humanoid_175_v2_tpose.npy",
        help="Path to save the generated T-pose npy.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the generated T-pose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = (script_dir / xml_path).resolve()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (script_dir / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not xml_path.exists():
        raise FileNotFoundError(f"Target XML not found: {xml_path}")

    skeleton = SkeletonTree.from_mjcf(str(xml_path))
    zero_pose = SkeletonState.zero_pose(skeleton)
    zero_pose.to_file(str(output_path))

    print(f"Saved v2 T-pose to: {output_path}")
    print(f"Loaded XML: {xml_path}")
    print(f"Skeleton nodes: {len(skeleton.node_names)}")

    if args.visualize:
        plot_skeleton_state(zero_pose)


if __name__ == "__main__":
    main()
