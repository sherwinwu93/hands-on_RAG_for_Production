from pathlib import Path
import pymupdf

pdf_path = Path(__file__).parent / "table.pdf"

with pymupdf.open(pdf_path) as document:
    for page_number, page in enumerate(document, start=1):
        # 在这一页中查找表格
        tables = page.find_tables()

        print(f"第{page_number}页找到{len(tables.tables)} 张表")
        for table_number, table in enumerate(tables.tables, start=1):
            rows = table.extract()

            print(f"\n表格{table_number}:")

            # 打印表格每一行
            # for row in rows:
            #     print(row)

            headers = rows[0]

            for row in rows[1:]:
                parts = []

                # 把headers和row对应起来,把header和cell对应起来
                for header, value in zip(headers, row):
                    parts.append(f"{header}: {value}")

                print(";".join(parts))