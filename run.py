import os
import base64
import json
from pillow_heif import register_heif_opener   # 新增
register_heif_opener()
from io import BytesIO
from PIL import Image
from openai import OpenAI

# ---------- 配置 ----------
FOLDER = "photos"
OUTPUT_DIR = "output"
CLASSIFY_BATCH = 4
MAX_GROUP_SIZE = 6
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"}

MAX_SIDE = 1680        # 压缩后图片长边最大像素
JPEG_QUALITY = 85      # JPEG 质量，85 在清晰度和体积间比较平衡

def get_api_key():
    with open("api_key.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

client = OpenAI(api_key=get_api_key(), base_url="https://api.deepseek.com")

# ---------- 图片预处理 + 编码 ----------
def encode_image(path):
    """读取图片 → 压缩 → 转 JPEG → base64 编码"""
    with Image.open(path) as img:
        # 处理透明通道（PNG 转 JPEG 前需去掉 alpha）
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 按长边缩放
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # 输出到内存，编码为 JPEG
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        data = buf.getvalue()

    return base64.b64encode(data).decode("utf-8")

def get_mime(path):
    """预处理后统一是 JPEG，无需再判断原格式"""
    return "jpeg"

def build_content(image_paths, instruction):
    content = [{"type": "text", "text": instruction}]
    for p in image_paths:
        fname = os.path.basename(p)
        content.append({"type": "text", "text": f"\n[图片文件名: {fname}]\n"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p)}"}
        })
    return content

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ---------- 阶段1：分类 ----------
def classify_batch(image_paths):
    instruction = """下面是一批课堂图片（板书或PPT）。请判断每张图片的内容主题（例如课程名+章节名，如"高等数学-导数"、"大学物理-力学"）。
要求：
- 主题名称要简短，同一门课的命名保持统一
- 返回严格的 JSON 对象，格式：{"文件名1.jpg": "主题A", "文件名2.jpg": "主题B"}
- 只输出 JSON，不要用 markdown 代码块包裹，不要输出其他内容"""
    content = build_content(image_paths, instruction)
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": content}]
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except Exception:
        print(f"  ⚠ 分类结果解析失败，输出原文：{text[:200]}")
        return {os.path.basename(p): "未分类" for p in image_paths}

# ---------- 阶段2：整合 ----------
def organize_group(image_paths, topic):
    instruction = f"""以下图片都属于同一主题：「{topic}」。
请阅读所有图片，把它们整合成一份完整的复习笔记。
要求：
1. 提取所有文字，修正明显的识别错误
2. 数学公式用 LaTeX（$...$ 或 $$...$$）
3. 表格用 Markdown 表格
4. 按知识点的逻辑结构重新组织，不要机械地按图片顺序罗列
5. 每部分标注来源图片文件名
6. 输出务必精炼，只保留核心知识点，不要展开解释，不要重复。
直接输出 Markdown 格式的笔记。"""
    content = build_content(image_paths, instruction)
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": content}]
    )
    result = resp.choices[0].message.content

    safe = "".join(c for c in topic if c.isalnum() or c in "_- ")
    safe = safe or "未分类"
    out_path = os.path.join(OUTPUT_DIR, f"{safe}.md")
    i = 1
    while os.path.exists(out_path):
        out_path = os.path.join(OUTPUT_DIR, f"{safe}_{i}.md")
        i += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  ✓ 已生成: {os.path.basename(out_path)}")

# ---------- 主流程 ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FOLDER):
        print(f"找不到 '{FOLDER}' 文件夹")
        return
    images = [os.path.join(FOLDER, f) for f in os.listdir(FOLDER)
              if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
    if not images:
        print(f"'{FOLDER}' 里没有图片")
        return
    print(f"共找到 {len(images)} 张图片\n")

    # === 阶段1：分类 ===
    topic_map = {}
    if len(images) <= 3:
        print("【阶段1】图片较少，跳过分类，直接合并处理\n")
        for p in images:
            topic_map[os.path.basename(p)] = "全部笔记"
    else:
        print("【阶段1】识别图片主题...")
        for i, batch in enumerate(chunk(images, CLASSIFY_BATCH)):
            print(f"  分析第 {i+1} 批（{len(batch)} 张）...")
            topic_map.update(classify_batch(batch))
        print()

    # === 按主题分组 ===
    groups = {}
    for fname, topic in topic_map.items():
        full = os.path.join(FOLDER, fname)
        if os.path.exists(full):
            groups.setdefault(topic, []).append(full)

    print(f"【阶段2】按主题整理，共 {len(groups)} 个主题：")
    for t, ps in groups.items():
        print(f"  · {t}（{len(ps)} 张）")
    print()

    # === 阶段2：逐主题整合 ===
    for topic, paths in groups.items():
        print(f"正在整理「{topic}」...")
        for sub in chunk(paths, MAX_GROUP_SIZE):
            organize_group(sub, topic)

    print(f"\n全部完成！结果在 {OUTPUT_DIR} 文件夹。")

if __name__ == "__main__":
    main()
