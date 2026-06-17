import uuid
import base64

def save_bytes_image(image_bytes, image_format="png"):
    id = uuid.uuid1()
    with open(f"{id}.{image_format}", "wb") as img_file:
        img_file.write(image_bytes)


def save_base64_image(base64_string):
    """
    将 Base64 字符串保存为图片文件
    """
    id = uuid.uuid1()
    # 1. 如果字符串包含 data URL 前缀（如 data:image/png;base64,），先去掉
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    # 2. Base64 解码为二进制数据
    image_data = base64.b64decode(base64_string)

    # 3. 写入文件
    with open(f"{id}.png", 'wb') as f:
        f.write(image_data)
