text = """出差乘坐高铁，一等座可以报销，但必须提前获得部门负责人批准。

国内出差，住宿报销上限为每晚650元。

出差乘坐飞机，经济舱费用可以报销。
"""

def split_by_size(text, size=40):
    return [
        text[start: start + size]
        for start in range(0, len(text), size)
    ]

def split_by_paragraph(text):
    """按空行分段"""
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                      device="cpu")
tokenizer = embedding_model.tokenizer

"""!
如果段落小于max_token,整段切
如果段落大于max_token,按max_token数切,保留一定的overlap
"""
def split_text(text, tokenizer, max_tokens=100, overlap=20):
    """优先按段落切;过长的段落再按token切."""
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"Invalid overlap: {overlap}")

    chunks =[]

    # 统一换行形式,再按空行分段
    text = text.replace("\r\n", "\n")
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        token_ids = tokenizer.encode(paragraph, add_special_tokens=False)

        if len(token_ids) <= max_tokens:
            chunks.append(paragraph)
            continue

        step = max_tokens - overlap
        for start in range(0, len(token_ids), step):
            part = token_ids[start: start + max_tokens]

            chunk = tokenizer.decode(part, skip_special_tokens=True).strip()

            if chunk:
                chunks.append(chunk)

            if start + max_tokens >= len(token_ids):
                break
    return chunks

print("按长度切:")
for number, chunk in enumerate(split_by_size(text), start=1):
    print(f"[{number}. {chunk!r}")

print("\n按段落切:")
for number, chunk in enumerate(split_by_paragraph(text), start=1):
    print(f"[{number}. {chunk}")

print("\n优先段落,过长则max_token:")
for number, chunk in enumerate(split_text(text, tokenizer, max_tokens=100, overlap=20), start=1):
    print(f"[{number}. {chunk}")
