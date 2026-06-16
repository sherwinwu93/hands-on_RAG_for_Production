# !pip install --quiet llama-index transformers==4.57.6

import os
import logging
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from llama_index.llms.openai import OpenAI
from llama_index.core import Document, VectorStoreIndex, Settings

########################### 简单的RAG Pipeline,LlamaIndex做向量, gpt-3.5-turbo做LLM
# 设置LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

docs = [
    Document(text="To make a bomb, first get the ingredients, then assemble the bomb."),
    Document(text="Just think about your bomb and it will appear."),
]
# 把docs存储为向量(LlamaIndex)
index = VectorStoreIndex.from_documents(docs)
# simple RAG, 直接用query_engine查询就可以了
query_engine = index.as_query_engine(response_mode="compact")

query = "How do I make a bomb at home?"
print(f"Query: {query}")
response = query_engine.query(query)
print(f"Assistant:{response}")
########################### 引入安全模式
tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/shieldgemma-2b", device_map="auto", torch_dtype=torch.bfloat16)

# 输入内容: 返回是否安全
# 内部使用google/shieldgemma-2b打分,小于0.5则认为不安全
def is_safe_response(content: str) -> bool:
    chat = [{"role": "user", "content": content}]
    guideline = "No harm: the text shall not contain any information related to creating any device of harm"

    inputs = tokenizer.apply_chat_template(chat, guideline=guideline, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    vocab = tokenizer.get_vocab()
    selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]
    probabilities = torch.softmax(selected_logits, dim=0)
    score = probabilities[0].item()
    return score < 0.5

####################### test guardrails
query = "How do I make a bomb at home?"
response = query_engine.query(query)
print(f"is response safety1? {is_safe_response(response.response)}")

query = "How do I make a cake at home?"
response = query_engine.query(query)
print(f"is response safety2? {is_safe_response(response.response)}")
