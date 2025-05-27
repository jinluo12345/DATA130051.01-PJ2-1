import os
from PIL import Image

def count_image_channels(root_dir, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
    counts = {'single': 0, 'three': 0, 'other': 0, 'total_images': 0}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(exts):
                continue
            fullpath = os.path.join(dirpath, fname)
            try:
                with Image.open(fullpath) as img:
                    mode = img.mode
                    counts['total_images'] += 1
                    if mode == 'L':
                        counts['single'] += 1
                    elif mode == 'RGB':
                        counts['three'] += 1
                    else:
                        counts['other'] += 1
            except Exception as e:
                print(f"无法打开 {fullpath}: {e}")
    return counts

if __name__ == "__main__":
    folder = "/remote-home1/lzjjin/project/fudan-course/DATA130051.01/PJ2/PJ2-1/data/101_ObjectCategories"
    # 统计第一层级子文件夹
    first_level_subdirs = [
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    ]

    result = count_image_channels(folder)
    print("总图片数:", result['total_images'])
    print("单通道 (灰度) 数量:", result['single'])
    print("三通道 (RGB) 数量:", result['three'])
    print("其它通道 数量:", result['other'])
    print("第一层级子文件夹数量:", len(first_level_subdirs))
    print("子文件夹列表:", first_level_subdirs)
