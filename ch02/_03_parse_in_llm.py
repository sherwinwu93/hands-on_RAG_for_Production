from openai import OpenAI

from config import set_environment

set_environment()

client = OpenAI()
file = client.files.create(
    file=open("sample_data/sample_data.pdf", "rb"),
    purpose="user_data"
)
model = "gpt-5.1"
#################### 提取段落文字
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    }
                },
                {
                    "type": "text",
                    "text": "Extract the text content from the file. Exclude texts from tables or images.",
                },
            ]
        }
    ]
)

print(completion.choices[0].message.content)
#################### 提取表格文字
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    }
                },
                {
                    "type": "text",
                    "text": "Extract the tables from the file. Return in Markdown tables.",
                },
            ]
        }
    ]
)

print(completion.choices[0].message.content)
#################### 提取图片
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    }
                },
                {
                    "type": "text",
                    # 以base64格式输出
                    "text": "Extract the image from the PDF file. Return as base64-encoded string. Only the base64 string, no other text.",
                },
            ]
        }
    ]
)

# 以base64格式输出
print(completion.choices[0].message.content[:256])