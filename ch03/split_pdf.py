# PDF Splitting for Large Documents
# !pip install --quiet PyPDF2
import os
import requests
import io
import math
from urllib.parse import urlparse
from PyPDF2 import PdfReader, PdfWriter

#################### PDF Splitting Functions
# Download PDF
# calculates chunks number
# PDF to PDFs
def get_pdf_reader(input_source):
    """
    Args:
        input_source: url of the PDF
    Returns:
        tuple: (PdfReader object, base filename, total pages)
    """
    base_filename = "output"
    response = requests.get(input_source, stream=True, timeout=30)
    # validate status code
    response.raise_for_status()

    parsed_url = urlparse(input_source)
    path_part = os.path.basename(parsed_url.path)
    if path_part and "." in path_part:
        base_filename = os.path.splitext(path_part)[0]

    pdf_content = io.BytesIO(response.content)
    reader = PdfReader(pdf_content)
    total_pages = len(reader.pages)
    print(f"Download successfully. Total pages: {total_pages}")
    return reader, base_filename, total_pages

def split_pdf(input_source, out_put_dir, pages_per_chunk):
    """
    Args:
        input_source: PDF url
        out_put_dir: directory of output
        pages_per_chunk
    return:
        multiple smaller PDF
    """
    reader, base_filename, total_pages = get_pdf_reader(input_source)

    if reader is None:
        print("failed to get PDF. Aborting")
        return

    try:
        os.makedirs(out_put_dir, exist_ok=True)

        # calculate chunks number
        num_chunks = math.ceil(total_pages / pages_per_chunk)
        print(f"num_chunks: {num_chunks}, page_per_chunk: {pages_per_chunk}")

        # process each chunk
        for i in range(num_chunks):
            writer = PdfWriter()
            start_page = i * pages_per_chunk
            end_page = min(start_page + pages_per_chunk, total_pages)

            print(f"i: {i}, startPage: {start_page}, end_page: {end_page}")

            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            output_filename = os.path.join(out_put_dir, f"{base_filename}_chunk_{i + 1}.pdf")

            with open(output_filename, 'wb') as outfile:
                writer.write(outfile)
            print(f"chunk {i} saved as {output_filename}")
        print("splitting completed")
    except Exception as e:
        print(f"Error: {e}")