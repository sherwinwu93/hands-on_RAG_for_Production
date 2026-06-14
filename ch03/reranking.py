# Reranking Retrieved Results
# !pip install -U sentence-transformers --quiet
from sentence_transformers.cross_encoder import CrossEncoder

#################### Prepare query and documents
query = "What is the main benefit of using a transformer model in NLP?"

documents = [
    "Recurrent Neural Networks (RNNs) were previously popular for sequence tasks.", # Less relevant
    "Transformers allow for parallel processing of input tokens, leading to faster training times compared to RNNs.", # Highly relevant (speed/parallelism)
    "BERT, a popular transformer model, achieves state-of-the-art results on many NLP benchmarks.", # Relevant context, but not the *main benefit* of the architecture itself
    "The attention mechanism in transformers enables the model to weigh the importance of different words in the input sequence.", # Highly relevant (attention mechanism)
    "Convolutional Neural Networks (CNNs) are primarily used in computer vision.", # Irrelevant
    "A key advantage of the transformer architecture is its ability to handle long-range dependencies more effectively than RNNs.", # Highly relevant (long-range dependencies)
    "You can fine-tune pre-trained transformer models for specific downstream tasks." # Related benefit, but maybe not the *main* architectural one
]

model_name = "BAAI/bge-reranker-v2-m3"
try:
    # will download if non-exists
    model = CrossEncoder(model_name)
    print(f"Loaded reranker model: {model_name}")
    print("-" * 30)
except Exception as e:
    print(f"loaded reranker model error: {e}")
    exit()

#################### Score and rerank
sentence_pairs = [[query, doc] for doc in documents]

print("calculating relevance scores...")
scores = model.predict(sentence_pairs, show_progress_bar=True)
print("---------")
print(f"scores:{scores}")
print("---------")

docs_with_scores = list(zip(documents, scores))
reranked_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

print(f"Query: {query}")
print("----------------------Original order")
for i, doc in enumerate(documents):
    print(f"{i+1}: {doc}")

print("----------------------reranked order")
for i, (doc, score) in enumerate(reranked_docs):
    print(f"{i+1}: score: {score: .4f} - {doc}")
