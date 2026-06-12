from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

#################### 句子转换成向量
sentences = [
    "I am a happy person.",
    "I am a joyful person.",
    "I am a pessimistic person.",
    "I am not an optimistic person."
]
embeddings = model.encode(sentences)

#################### 数据库连接,pgvector插件安装,创建数据库,创建表,创建索引
# Database connection parameters
pg_password = os.environ.get('PGVECTOR_PASSWORD', 'RAGBOOK')
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': pg_password,
    'database': 'postgres'  # Connect to default database first
}

# Connect to PostgreSQL
conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = True
cursor = conn.cursor()

DB_FOR_VECTORS = "ragbook" # lowercase only, use a non-default database for our experiments

# Create a database if it doesn't exist
try:
    cursor.execute(f"CREATE DATABASE {DB_FOR_VECTORS};")
except psycopg2.errors.DuplicateDatabase:
    print (f'Database {DB_FOR_VECTORS} already exists.')

# Switch to the new database
DB_CONFIG['database'] = DB_FOR_VECTORS
conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = True
cursor = conn.cursor()

# Create a table for storing vectors

# Enable PG-Vector extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Drop the table if it already exists
cursor.execute("DROP TABLE IF EXISTS sentence_embeddings;")

# Create a table called `sentence_embeddings`
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentence_embeddings (
        text TEXT NOT NULL,
        embedding VECTOR(384)  -- all-MiniLM-L6-v2 produces 384-dimensional embeddings
    );
""")

# Create HNSW index in the table `sentence_embeddings` for efficient similarity search
cursor.execute("""
    CREATE INDEX 
    ON sentence_embeddings 
    USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 64);
""")
#################### 向数据库插入向量
# Insert sample sentences and their embeddings into the table `sentence_embeddings`

for sentence, embedding in zip(sentences, embeddings):
        embedding_as_list: List[float] = embedding.tolist()
        cursor.execute(
            "INSERT INTO sentence_embeddings (text, embedding) VALUES (%s, %s::vector)",
            (sentence, embedding_as_list)
        )
#################### 查询验证
cursor.execute("SELECT text, embedding FROM sentence_embeddings;")
rows = cursor.fetchall()
for row in rows:
    print(row)
#################### 从pgvector找出top-k(余铉相似度)
def vector_search(query, model, top_k):
    query_embedding = model.encode([query])[0]
    query_embedding_as_list: List[float] = query_embedding.tolist()

    cursor.execute(
            """
            SELECT text, 
                    1 - (embedding <=> %s::vector) as similarity
            FROM sentence_embeddings
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding_as_list, query_embedding_as_list, top_k)
        )

    results = cursor.fetchall()

    for row in results:
        print(row)

query = "I am a smiling person."
vector_search(query, model, 3)

query = "I have a bad feeling about the future."
vector_search(query, model, 3)

