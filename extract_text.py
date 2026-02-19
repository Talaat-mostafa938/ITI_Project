from langchain_community.document_loaders import PyPDFLoader , Docx2txtLoader
import os


def get_file_extension(file_path):
  _,extension = os.path.splitext(file_path)
  return extension[1:].lower()

def get_text_extraction(uploaded_file):
  
  ext = get_file_extension(uploaded_file)
  text = ""
    
  if ext == "pdf":
      loader = PyPDFLoader(uploaded_file)
            
  elif ext == "docx":
      loader = Docx2txtLoader(uploaded_file)
    
  pages = loader.load()
  text = "\n".join([page.page_content for page in pages])
  
  return text

