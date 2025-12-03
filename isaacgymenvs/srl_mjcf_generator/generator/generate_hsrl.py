import argparse
import os
from jinja2 import Environment, FileSystemLoader

# 当前文件所在目录（mjcf_generator/）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 构建绝对路径
BASE_XML_PATH = os.path.join(CURRENT_DIR, "../base/base_humanoid.xml")
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "../templates")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../output")

TEMPLATE_NAME = "SRL_template.xml.j2"

def generate_hsrl_model(
    leg1_length=0.60,
    leg2_length=0.55,
    enable_freejoint_z=1,
    enable_freejoint_y=1,
    enable_freejoint_x=0,
    base_width=0.095,
    base_distance=0.60,
    output_name="humanoid_with_srl.xml",
    output_dir=None,       
):
    """
    生成 humanoid + SRL 模型 XML 

    参数:
        leg1_length (float)
        leg2_length (float)
        enable_freejoint_z, enable_freejoint_y, enable_freejoint_x
        base_width (float)
        base_distance (float)
        output_name (str): 输出文件名
        output_dir (str):  XML 输出目录（若为 None, 则使用默认 OUTPUT_DIR)

    返回:
        out_path (str): 生成的 XML 文件绝对路径
    """

    try:
        # 读取基础 humanoid XML
        with open(BASE_XML_PATH, "r") as f:
            base_xml = f.read()

        # 渲染 Jinja2 模板
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template(TEMPLATE_NAME)

        srl_xml = template.render(
            leg1_length=leg1_length,
            leg2_length=leg2_length,
            enable_freejoint_z=enable_freejoint_z,
            enable_freejoint_y=enable_freejoint_y,
            enable_freejoint_x=enable_freejoint_x,
            base_width=base_width,
            leg_y_offset=base_width + 0.035,
            base_distance=base_distance
        )

        # 插入 SRL 子树
        final_xml = base_xml.replace("<!-- SRL_INSERT_HERE -->", srl_xml)

        # ---- 选择输出目录 ----
        if output_dir is None:
            output_dir = OUTPUT_DIR   # 旧逻辑：使用默认目录

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # ---- 拼接输出文件路径 ----
        if not output_name.endswith(".xml"):
            output_name += ".xml"

        out_path = os.path.join(output_dir, output_name)

        # ---- 写入 XML 文件 ----
        with open(out_path, "w") as f:
            f.write(final_xml)

        return out_path

    except Exception as e:
        print(f"[ERROR] SRL model generation failed: {e}")
        return None

def generate_srl_xml(leg1_length, leg2_length, enable_freejoint_z, enable_freejoint_y,
                     enable_freejoint_x, base_width, base_distance):
    """渲染 SRL 子模型 XML"""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_NAME)

    return template.render(
        leg1_length=leg1_length,
        leg2_length=leg2_length,
        enable_freejoint_z=enable_freejoint_z,
        enable_freejoint_y=enable_freejoint_y,
        enable_freejoint_x=enable_freejoint_x,
        base_width=base_width,
        leg_y_offset=base_width + 0.035,
        base_distance=base_distance
    )


def insert_srl(base_xml, srl_xml):
    """将 SRL 子树插入 base humanoid"""
    return base_xml.replace("<!-- SRL_INSERT_HERE -->", srl_xml)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--leg1", type=float, default=0.6 )
    parser.add_argument("--leg2", type=float, default=0.55 )
    parser.add_argument("--enable_freejoint_z", type=int, default=1)
    parser.add_argument("--enable_freejoint_y", type=int, default=1)
    parser.add_argument("--enable_freejoint_x", type=int, default=0)
    parser.add_argument("--base_width", type=float, default=0.095)
    parser.add_argument("--base_distance", type=float, default=0.60)
    parser.add_argument("--out", type=str, default="humanoid_with_srl.xml")

    args = parser.parse_args()

    print("Generating SRL MJCF:")
    print(f" - leg1_length:        {args.leg1}")
    print(f" - leg2_length:        {args.leg2}")
    print(f" - enable_freejoint_z: {args.enable_freejoint_z}")
    print(f" - enable_freejoint_y: {args.enable_freejoint_y}")
    print(f" - enable_freejoint_x: {args.enable_freejoint_x}")
    print(f" - base_width:         {args.base_width}")
    print(f" - base_distance:      {args.base_distance}")

    with open(BASE_XML_PATH, "r") as f:
        base_xml = f.read()

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_NAME)

    srl_xml = template.render(
        leg1_length=args.leg1,
        leg2_length=args.leg2,
        enable_freejoint_z=args.enable_freejoint_z,
        enable_freejoint_y=args.enable_freejoint_y,
        enable_freejoint_x=args.enable_freejoint_x,
        base_width=args.base_width,
        leg_y_offset=args.base_width + 0.035,
        base_distance=args.base_distance
    )

    final_xml = insert_srl(base_xml, srl_xml)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, args.out)

    with open(out_path, "w") as f:
        f.write(final_xml)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
