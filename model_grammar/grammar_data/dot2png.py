# _*_ coding utf-8 _*_
# Designer: はなちゃん
# Time: 2022/1/29 19:20
# Name: dot2png.py
from pathlib import Path
import subprocess
 
 
def check_valid_path(path):
    """检查文件路径是否有效，并返回以'/'分割的文件路径"""
    if '\\' in path:
        elements = str(Path(path)).split(sep='\\')
        final_path = Path('/'.join(elements))
    elif '/' in path:
        final_path = Path(path)
    else:
        if not Path(path).exists():
            raise Exception("Error: File path pattern is not correct.")
        else:
            final_path = Path(path)
    if not final_path.exists():
        raise Exception("Error: Your file path does not exist.")
    if final_path != '':
        return final_path
    else:
        return None
 
 
def dot2png(dot_file_path=None, img_path=None):
    """决策树可视化中.dot文件转化为.png图片的函数"""
    if not dot_file_path:
        raise Exception(".dot file is not given.")
    elif not dot_file_path.endswith('.dot'):
        raise Exception("file provided is not '.dot' type.")
 
    DOT_PATH = check_valid_path(dot_file_path)
 
    if not img_path:
        img_path = 'dt_png.png'
    elif not img_path.endswith('.png'):
        raise Exception("image file not end with '.png'.")
 
    IMG_PATH = img_path
 
    cmd_args = ['dot', '-Tpng', DOT_PATH, '-o', IMG_PATH]
 
    cmd_pro = subprocess.Popen(args=cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = cmd_pro.stdout.read().decode('gbk')
    if retval == '':
        print("successfully create file " + IMG_PATH)
    else:
        print("The program encountered some error: ")
        print(retval)

if __name__ == '__main__':
    dot2png('grammar_apr30_wheel.dot','grammar_apr30_wheel.png')