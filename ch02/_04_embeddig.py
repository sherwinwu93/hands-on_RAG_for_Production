from sentence_transformers import SentenceTransformer

######################################
# 1. 用向量模型把句子转成向量
# 2. 向量与向量计算余铉相似度
# 向量模型
model = SentenceTransformer('all-MiniLM-L6-v2')
#################### 生成向量
# List of sentences to encode
sentences = [
    "I am a happy person.",
    "I am a joyful person.",
    "I am a pessimistic person.",
    "I am not an optimistic person."
]
embeddings = model.encode(sentences)
print(embeddings.shape)
print(embeddings[:, :5])

#################### 余铉相似度方针
# compute the cosine similarity between the embeddings
similarity_matrix = model.similarity(embeddings, embeddings)
print(similarity_matrix)
