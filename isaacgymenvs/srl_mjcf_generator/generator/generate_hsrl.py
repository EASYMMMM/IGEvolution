import argparse
import os
from typing import Optional
from jinja2 import Environment, FileSystemLoader

# 当前文件所在目录（例如 mjcf_generator/）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认路径（可通过命令行覆盖）
BASE_XML_PATH = os.path.join(CURRENT_DIR, "../base/base_real_hri.xml")
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "../templates")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../output")
TEMPLATE_NAME = "SRL_template_real_hri_v1.xml.j2"
INSERT_MARKER = "<!-- SRL_INSERT_HERE -->"


def build_jinja_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )


def render_srl_template(
    leg1_length: float,
    leg2_length: float,
    enable_freejoint_z: int,
    enable_freejoint_y: int,
    enable_freejoint_x: int,
    base_width: float,
    base_distance: float,
    template_dir: str = TEMPLATE_DIR,
    template_name: str = TEMPLATE_NAME,
) -> str:
    """渲染 SRL 子模型 XML。仅传入联合优化所需的原始 7 个设计变量。"""
    env = build_jinja_env(template_dir)
    template = env.get_template(template_name)
    return template.render(
        leg1_length=leg1_length,
        leg2_length=leg2_length,
        enable_freejoint_z=int(enable_freejoint_z),
        enable_freejoint_y=int(enable_freejoint_y),
        enable_freejoint_x=int(enable_freejoint_x),
        base_width=base_width,
        base_distance=base_distance,
    )


def insert_srl(base_xml: str, srl_xml: str, marker: str = INSERT_MARKER) -> str:
    """将 SRL 子树插入 base humanoid。"""
    if marker not in base_xml:
        raise ValueError(f"Insert marker not found in base XML: {marker}")
    return base_xml.replace(marker, srl_xml, 1)


def generate_hsrl_model(
    leg1_length: float = 0.60,
    leg2_length: float = 0.55,
    enable_freejoint_z: int = 1,
    enable_freejoint_y: int = 1,
    enable_freejoint_x: int = 0,
    base_width: float = 0.095,
    base_distance: float = 0.60,
    output_name: str = "humanoid_with_srl.xml",
    output_dir: Optional[str] = None,
    base_xml_path: str = BASE_XML_PATH,
    template_dir: str = TEMPLATE_DIR,
    template_name: str = TEMPLATE_NAME,
) -> Optional[str]:
    """
    生成 humanoid + SRL 模型 XML。

    参数:
        leg1_length (float)
        leg2_length (float)
        enable_freejoint_z, enable_freejoint_y, enable_freejoint_x (int)
        base_width (float)
        base_distance (float)
        output_name (str): 输出文件名
        output_dir (Optional[str]): XML 输出目录（None 时使用默认 OUTPUT_DIR）
        base_xml_path (str): base XML 路径
        template_dir (str): 模板目录
        template_name (str): 模板文件名

    返回:
        out_path (Optional[str]): 生成的 XML 文件绝对路径
    """
    try:
        with open(base_xml_path, "r", encoding="utf-8") as f:
            base_xml = f.read()

        srl_xml = render_srl_template(
            leg1_length=leg1_length,
            leg2_length=leg2_length,
            enable_freejoint_z=enable_freejoint_z,
            enable_freejoint_y=enable_freejoint_y,
            enable_freejoint_x=enable_freejoint_x,
            base_width=base_width,
            base_distance=base_distance,
            template_dir=template_dir,
            template_name=template_name,
        )

        final_xml = insert_srl(base_xml, srl_xml)

        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if not output_name.endswith(".xml"):
            output_name += ".xml"

        out_path = os.path.join(output_dir, output_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_xml)

        return out_path

    except Exception as e:
        print(f"[ERROR] SRL model generation failed: {e}")
        return None


def generate_srl_xml(
    leg1_length: float,
    leg2_length: float,
    enable_freejoint_z: int,
    enable_freejoint_y: int,
    enable_freejoint_x: int,
    base_width: float,
    base_distance: float,
    template_dir: str = TEMPLATE_DIR,
    template_name: str = TEMPLATE_NAME,
) -> str:
    """向后兼容接口：仅渲染 SRL 子模型 XML。"""
    return render_srl_template(
        leg1_length=leg1_length,
        leg2_length=leg2_length,
        enable_freejoint_z=enable_freejoint_z,
        enable_freejoint_y=enable_freejoint_y,
        enable_freejoint_x=enable_freejoint_x,
        base_width=base_width,
        base_distance=base_distance,
        template_dir=template_dir,
        template_name=template_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate humanoid + SRL MJCF from base XML and Jinja2 template.")

    parser.add_argument("--leg1", type=float, default=0.60, help="Upper-leg collision length (leg1_length)")
    parser.add_argument("--leg2", type=float, default=0.55, help="Lower-leg collision length (leg2_length)")
    parser.add_argument("--enable_freejoint_z", type=int, default=1)
    parser.add_argument("--enable_freejoint_y", type=int, default=1)
    parser.add_argument("--enable_freejoint_x", type=int, default=0)
    parser.add_argument("--base_width", type=float, default=0.095)
    parser.add_argument("--base_distance", type=float, default=0.60)
    parser.add_argument("--out", type=str, default="humanoid_with_srl_real_hri.xml")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--base_xml", type=str, default=BASE_XML_PATH)
    parser.add_argument("--template_dir", type=str, default=TEMPLATE_DIR)
    parser.add_argument("--template_name", type=str, default=TEMPLATE_NAME)

    args = parser.parse_args()

    print("Generating SRL MJCF with new real_hri template:")
    print(f" - base_xml:           {args.base_xml}")
    print(f" - template_name:      {args.template_name}")
    print(f" - leg1_length:        {args.leg1}")
    print(f" - leg2_length:        {args.leg2}")
    print(f" - enable_freejoint_z: {args.enable_freejoint_z}")
    print(f" - enable_freejoint_y: {args.enable_freejoint_y}")
    print(f" - enable_freejoint_x: {args.enable_freejoint_x}")
    print(f" - base_width:         {args.base_width}")
    print(f" - base_distance:      {args.base_distance}")
    print(f" - output_dir:         {args.output_dir}")
    print(f" - output_name:        {args.out}")

    out_path = generate_hsrl_model(
        leg1_length=args.leg1,
        leg2_length=args.leg2,
        enable_freejoint_z=args.enable_freejoint_z,
        enable_freejoint_y=args.enable_freejoint_y,
        enable_freejoint_x=args.enable_freejoint_x,
        base_width=args.base_width,
        base_distance=args.base_distance,
        output_name=args.out,
        output_dir=args.output_dir,
        base_xml_path=args.base_xml,
        template_dir=args.template_dir,
        template_name=args.template_name,
    )

    if out_path is None:
        raise SystemExit(1)

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
