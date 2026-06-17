# pip install langchain langchain-openai faiss-cpu redis
import os
import redis
import hashlib
import json
import numpy as np
from typing import List, Optional, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import PrivateAttr

from config import set_environment

set_environment()

############ 启动redis docker,并连接redis
import subprocess
import time
# Remove existing container if present, then start fresh
subprocess.run(["docker", "rm", "-f", "redis-cache"], capture_output=True)
subprocess.run(["docker", "run", "-d", "--name", "redis-cache", "-p", "6379:6379", "redis"], check=True)

# Wait for Redis to be ready
time.sleep(2)
print("Redis container started.")

cache = redis.Redis(host="localhost", port=6379)

############## 1. 绝对匹配缓存
class CachedRetriever(BaseRetriever):
    """用Redis包装BaseRetriever,使其有query-结果缓存能力"""
    _base_retriever: any = PrivateAttr()
    _cache_ttl: int = PrivateAttr(default=3600)

    def __init__(self, base_retriever, cache_ttl: int = 3600, **kwargs):
        super().__init__(**kwargs)
        self._base_retriever = base_retriever
        self._cache_ttl = cache_ttl

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. 把Query Hash
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()

        # 2. 从缓存拿数据
        cached_result = cache.get(query_hash)
        if cached_result:
            print("Cache Hit! Skipping vector search.")
            docs_data = json.loads(cached_result)
            return [Document(page_content=d['content'], metadata=d['metadata']) for d in docs_data]

        # 3. 缓存没有数据,则执行向量搜索
        print("Cache Miss. Performing vector search...")
        results = self._base_retriever.invoke(query)

        # 4. 缓存向量数据
        docs_data = [{'content': doc.page_content, 'metadata': doc.metadata} for doc in results]
        cache.setex(query_hash, self._cache_ttl, json.dumps(docs_data))
        return results
##################### 2. 语义缓存
# 缓存 向量-结果
# 新query转换成向量,计算和缓存键的相似度
# 如果相似度大于阈值,则返回缓存结果,否则执行向量搜索
class SemanticCachedRetriever(BaseRetriever):
    """用Redis包装BaseRetriever,使其有向量-结果缓存能力"""
    _base_retriever: any = PrivateAttr()
    _embeddings: any = PrivateAttr()
    _similarity_threshold: float = PrivateAttr(default=0.85)
    _cache_embeddings: List[np.ndarray] = PrivateAttr(default_factory=list)
    _cache_results: List[List[Document]] = PrivateAttr(default_factory=list)
    _cache_queries: List[str] = PrivateAttr(default_factory=list)

    def __init__(
            self,
            base_retriever,
            embeddings,
            similarity_threshold: float = 0.85,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._base_retriever = base_retriever
        self._embeddings = embeddings
        self._similarity_threshold = similarity_threshold
        self._cache_embeddings = []
        self._cache_results = []
        self._cache_queries = []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # 计算两个向量的余铉相似度
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _find_similar_cached(
            self, query_embedding: np.ndarray
    ) -> Optional[Tuple[List[Document], str, float]]:
        """找到相似的缓存.
        这里的实现是拿所有键去计算相似度, 可以用FAISS(支持百万级别)
        """
        best_similarity = 0.0
        best_match = None
        best_query = None

        for i, cached_emb in enumerate(self._cache_embeddings):
            similarity = self._cosine_similarity(query_embedding, cached_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = self._cache_results[i]
                best_query = self._cache_queries[i]

        if best_similarity >= self._similarity_threshold:
            return best_match, best_query, best_similarity
        return None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 把query转换成向量
        query_embedding = np.array(self._embeddings.embed_query(query))

        # 用query的向量找缓存命中
        cache_hit = self._find_similar_cached(query_embedding)
        if cache_hit:
            docs, original_query, similarity = cache_hit
            print(f"Semantic Cache Hit! (similarity: {similarity:.3f})")
            print(f"  Matched query: \"{original_query}\"")
            return docs

        # 缓存未命中,则执行向量搜索
        print("Semantic Cache Miss. Performing vector search...")
        results = self._base_retriever.invoke(query)

        # 缓存查询向量和结果
        self._cache_embeddings.append(query_embedding)
        self._cache_results.append(results)
        self._cache_queries.append(query)

        return results

    def clear_cache(self):
        """Clear the semantic cache."""
        self._cache_embeddings = []
        self._cache_results = []
        self._cache_queries = []
################## 0. 准备文档
sample_text = """
Artificial Intelligence (AI) is transforming industries worldwide.
Machine learning models can analyze vast amounts of data to find patterns.
Deep learning uses neural networks with multiple layers to learn representations.
Natural Language Processing (NLP) enables computers to understand human language.
Retrieval-Augmented Generation (RAG) combines retrieval systems with language models.
RAG helps reduce hallucinations by grounding responses in retrieved documents.
Vector databases store embeddings for efficient similarity search.
Caching can significantly improve RAG system performance by avoiding redundant searches.
"""
with open("sample_docs.txt", "w") as f:
    f.write(sample_text)
print("文件写入")

################## 1. 切块
# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# Load and split documents
loader = TextLoader("sample_docs.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

print(f"Created {len(texts)} text chunks")
################## 2. 存储为向量,并初始化CachedRetriever
vectorstore = FAISS.from_documents(texts, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

################################################################### 测试普通缓存
# Wrap with caching
cached_retriever = CachedRetriever(base_retriever, cache_ttl=3600)

print("Vector store and cached retriever initialized.")
################## 3. RAG chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": cached_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain created with cached retriever.")
################## 4. 把缓存清空掉(避免影响测试)
query = "What is RAG and how does it help?"
query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
cache.delete(query_hash)

print("Cache cleared for demo query.")
################## 5. 第一次查询,无缓存
print("=== First Query (Cache Miss) ===")
response = rag_chain.invoke(query)
print(f"\nAnswer: {response}")
################## 6. 第二次查询,有缓存
print("=== Second Query (Cache Hit) ===")
response = rag_chain.invoke(query)
print(f"\nAnswer: {response}")
################################################################### 测试语义缓存
original_query = "What is RAG and how does it help?"
similar_query = "Tell me about RAG and its benefits"

_ = cached_retriever.invoke(original_query)
_ = cached_retriever.invoke(similar_query)

################## 7. semantic cached retriever
semantic_retriever = SemanticCachedRetriever(
    base_retriever,
    embeddings,
    similarity_threshold=0.85
)
_ = semantic_retriever.invoke(original_query)
_ = semantic_retriever.invoke(similar_query)
