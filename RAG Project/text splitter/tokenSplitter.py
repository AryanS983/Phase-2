from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader


loader = PyPDFLoader("./text splitter/docs/PRD8 Template_v3.0 - Google Docs.pdf")
docs = loader.load()


text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10) # 1st 1000 tokens of the 1st page of the docs

chunks = text_splitter.split_documents(docs)
print(len(chunks))