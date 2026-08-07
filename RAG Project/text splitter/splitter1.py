from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("./text splitter/docs/testing.txt")
pages = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=2,
    separator="\n"      # by default "/n /n"
)


# splitter = RecursiveCharacterTextSplitter(    # Recursively splits text based on separators ["\n\n", "\n", " ", ""]
#     chunk_size=10,
#     chunk_overlap=2,
#     separator="\n"      
# )

chunks = splitter.split_documents(pages)


print(list(map(lambda x: x.page_content, chunks)))