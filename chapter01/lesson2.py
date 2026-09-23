#################### 1.把文档切成小块
handbook = """
国内出差，住宿报销上限为每晚650元。

国内出差，餐费补贴为每天100元。

出差期间的打车费用，需要提供发票。

出差火车票可以报销二等座费用。

出差乘坐飞机，经济舱费用可以报销。
"""

# chunks: 切块, 按空格切分
chunks = [
    paragraph.strip()
    for paragraph in handbook.split('\n\n')
    if paragraph.strip()
]

for number, chunk in enumerate(chunks, start=1):
    print(f'D{number}: {chunk}')
#################### 2.让高铁找到火车票
keyword_groups = {
    "住宿": ["住宿", "酒店", "宾馆"],
    "餐费": ["餐费", "吃饭", "伙食"],
    "打车": ["打车", "出租车", "网约车"],
    "火车": ["火车", "高铁", "动车"],
    "飞机": ["飞机", "机票"],
}

def retrieve(question, chunks):
    """遍历keyword_groups的value,匹配的则放入results返回"""
    results = []

    for chunk in chunks:
        for topic, words in keyword_groups.items():
            question_match = any(word in question for word in words)

            if question_match and topic in chunk:
                results.append(chunk)
                break
    return results

question = input("\n你想问什么?")
results = retrieve(question, chunks)

if results:
    print("\n找到的资料:")
    for result in results:
        print(result)
else:
    print("没有找到相关资料。")
#################### 3. 向量解决了什么问题
# 用模型把文字转换成一串数字,意思接近的文字,数字更接近
# 两条流程: 入库: 文档->提取文字->切块->生成向量->报错向量~原文和来源
#         问答: 问题->生成问题向量->找相近的切块->把原文和问题交给模型->生成回答
