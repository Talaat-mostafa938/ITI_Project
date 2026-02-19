from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_vectorstore(text_chunks):

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")

    return vector_store
