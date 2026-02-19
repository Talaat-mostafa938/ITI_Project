from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)