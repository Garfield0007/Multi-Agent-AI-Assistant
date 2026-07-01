from sentence_transformers import SentenceTransformer
import chromadb
# ---------------- LOAD EMBEDDING MODEL ----------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#-----------------chromadb client-----------------
client = chromadb.Client()

#------------------create collection-----------------
collection = client.get_or_create_collection(name="knowledge_base")


# ---------------- DOCUMENTS ----------------

documents = [

    "ADK stands for Agent Development Kit used for building AI agents.",

    "RAG means Retrieval-Augmented Generation.",

    "Gemini is Google's family of AI models.",

    "Multi-agent systems contain multiple specialized AI agents."
]

#-----------------create embedding------------------
existing = collection.count()
if existing == 0:
    for index,document in enumerate(documents):
        embedding = embedding_model.encode(document).tolist()
        collection.add(
            documents=[document],
              embeddings=[embedding],
                ids=[str(index)]
    )

#-----------------search function-----------------
def search_knowledge_base(query: str):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    return results['documents'][0][0] 

#-----------------Test------------------

# question = "explain rag?"

# answer = search_knowledge_base(question)

# print(answer)

