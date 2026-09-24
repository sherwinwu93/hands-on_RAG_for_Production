from pathlib import Path
from pypdf import PdfReader

folder = Path(__file__).parent
reader = PdfReader(folder/'alice.pdf')

pages_text = []

"""!
把每页解析到的文字,添加到pages_text里面
"""
for number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    pages_text.append(f"============第{number}页=============\n{text}")

all_text = "\n\n".join(pages_text)

"""!
把pages_text的文字,保存到pdf_text.txt里面
"""
output_path = folder / "pdf_text.txt"
output_path.write_text(all_text, encoding="utf-8")

print("已保存到:", output_path)