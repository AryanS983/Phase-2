from langchain_docling.loader import DoclingLoader

FILE_PATH = "./document loaders/docs/emp.txt"

loader = DoclingLoader(file_path=FILE_PATH)

docs = loader.load()

print(docs[0].page_content)