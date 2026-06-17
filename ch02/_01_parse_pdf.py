# pdf文件的概念: 页~段落文字~表格文字~图片
# 方法:
#   提取Pdf的页
#   提取Pdf的文本(段落文字+表格文字)
#   提取表格文字
#   只提取段落文字(不提取表格文字)
#   提取图片
import json
import pymupdf
from IPython.display import Image, display
import io

from save_image import save_bytes_image


def extract_page(pdf_path):
    """提取Pdf的页"""
    # 加载文件,按页读取
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text: str = page.get_text()
            print(f"page{i}-----")
            print(text, end="---")
def extract_text_blocks(pdf_path):
    """提取Pdf的文本块(段落文字+表格文字)"""
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text: list = page.get_text("blocks")
            for i, block in enumerate(text):
                print(f"block{i+1}-----")
                print(block[4], end="---\n")

def extract_table_content(pdf_path):
    """提取Pdf的表格文字"""
    with pymupdf.open(pdf_path) as doc:
      for page in doc:
        tables = page.find_tables() # locate and extract any tables on page
        for i, table in enumerate(tables):
          print (f"Table {i+1}")
          print(table.extract(), end="\n\n")
def extract_unstructured_text(pdf_path):
    """只提取Pdf的段落文字(不提取表格)"""
    unstructured_text = []

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            # Get all text from the page
            page_text = page.get_text()

            # Find tables on the page to exclude their content
            tables = page.find_tables()

            # Get table text blocks to filter out
            table_text_blocks = []
            if tables.tables:
                for table in tables:
                    # Get table bounding box
                    bbox = table.bbox
                    # Extract text within table bounds
                    table_text = page.get_text(clip=bbox)
                    table_text_blocks.append(table_text.strip())

            # Split page text into lines and filter out table content
            lines = page_text.split('\n')
            filtered_lines = []

            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    # Check if this line is part of any table
                    is_table_content = False
                    for table_text in table_text_blocks:
                        if line in table_text:
                            is_table_content = True
                            break

                    # Only include non-table content
                    if not is_table_content:
                        filtered_lines.append(line)

            # Join filtered lines back into paragraphs
            page_unstructured = '\n'.join(filtered_lines)
            if page_unstructured.strip():
                unstructured_text.append(page_unstructured)

    return '\n\n'.join(unstructured_text)

def extract_images(pdf_path):
    """提取Pdf的图片"""
    with pymupdf.open(pdf_path) as doc:
      for page in doc:
        # Get images from the page
        image_list = page.get_images()
        for image_index, img in enumerate(image_list):
          # Extract image data
          xref = img[0]
          base_image = doc.extract_image(xref)
          image_bytes = base_image["image"]
          # Display the image directly in Jupyter Notebook
          display(Image(data=image_bytes))

          # save image
          image_format = base_image["ext"]
          save_bytes_image(image_bytes, image_format)


pdf_path = "sample_data/sample_data.pdf"
# extract_page(pdf_path)
# extract_text_blocks(pdf_path)
# extract_table_content(pdf_path)
# unstructured_content = extract_unstructured_text(pdf_path)
# print(f"Unstructured text extracted: {unstructured_content}")
extract_images(pdf_path)