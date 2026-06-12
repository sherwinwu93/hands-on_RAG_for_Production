########################################
# Load -> Split -> Embed & Store -> Query
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import LanceDB
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import set_environment

set_environment()

########## Step1: Load PDF
pdf_url = "https://www.adobe.com/be_en/active-use/pdf/Alice_in_Wonderland.pdf"
loader = PyPDFLoader(pdf_url)
pages = loader.load()

print(f"Loaded {len(pages)} pages")
########## Step2: Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
chunks = text_splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")
########## Step3: Embed & Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = LanceDB.from_documents(chunks, embeddings)

print(f"Created vector store with {len(chunks)} chunks")
########## Step4: RAG Query
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:

{context}

Question: {question}
    """
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Who is the main character and what happens at the beginning of the story?",
    "What did the Caterpillar ask Alice?",
    "Describe the Mad Hatter's tea party.",
    "What happened at the trial at the end of the story?"
]
for q in questions:
    answer = rag_chain.invoke(q)
    print(f"Q: {q}")
    print(f"A: {answer}")
    print("-" * 20)
