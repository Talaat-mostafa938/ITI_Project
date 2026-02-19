from fastapi import FastAPI
import os
import gradio as gr

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from extract_text import get_text_extraction
from chunk_text import get_text_chunks
from vectorstore import get_vectorstore

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

faiss_index = None
if os.path.exists("faiss_index"):
    faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

app = FastAPI(
    title="LangChain Server with Gradio UI",
    version="1.0",
    description="FastAPI server running LangServe and Gradio for PDF Chatbot",
)


def docs2str(docs):
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, "metadata", None).get("title") if getattr(doc, "metadata", None) else None
        if doc_name: 
            out_str += f"[Quote from {doc_name}]\n"
        out_str += getattr(doc, "page_content", str(doc)) + "\n"
    return out_str

def retrieve_documents(query: str):
    """دالة لجلب المستندات لو الـ Index موجود"""
    if faiss_index is None:
        return "No documents uploaded yet."
    docs = faiss_index.similarity_search(query, k=4)
    return docs2str(docs)

chat_prompt = ChatPromptTemplate.from_template(
    """
    You are a document chatbot. Help the user as they ask questions about documents.
    
    The following information may be useful for your response: 
    Document Retrieval:
    {context}
    
    Answer only from retrieval. Only cite sources that are used. Make your response conversational.
    
    Question: {input}
    """
)

def build_rag_chain(api_key: str):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    rag_chain = (
        {"context": RunnableLambda(retrieve_documents), "input": RunnablePassthrough()}
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def process_file(uploaded_file):
    global faiss_index
    
    if uploaded_file is None:
        return "Please upload a file first."
    
    try:
        text = get_text_extraction(uploaded_file.name)
        chunks = get_text_chunks(text)
        get_vectorstore(chunks) 
        
        faiss_index = FAISS.load_local("faiss_index", 
                                       embeddings, 
                                       allow_dangerous_deserialization=True)
        return "✅ File processed successfully! You can start asking questions."
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"


def chat_with_bot(message, history, api_key):
    global faiss_index
    
    if not api_key:
        return "⚠️ Please enter your GROQ API key."

    if faiss_index is None:
        return "⚠️ Please upload and process a document first."
    
    try:
        rag_chain = build_rag_chain(api_key)
        response = rag_chain.invoke(message)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"


with gr.Blocks(title="Chat with your Docs") as gradio_ui:
    gr.Markdown("# 📚 Chat with your PDF/DOCX Documents")

    with gr.Row():
        with gr.Column(scale=1):

            api_key_input = gr.Textbox(
                label="Enter GROQ API Key",
                type="password"
            )

            file_input = gr.File(label="Upload Document")
            process_btn = gr.Button("Process File", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

            process_btn.click(
                fn=process_file,
                inputs=file_input,
                outputs=status_text
            )

        with gr.Column(scale=2):

            chatbot = gr.ChatInterface(
                fn=chat_with_bot,
                additional_inputs=[api_key_input]
            )


app = gr.mount_gradio_app(app, gradio_ui, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9012)