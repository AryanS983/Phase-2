from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=2
)

vector = embeddings.embed_query("Kolkata")
vector1 = embeddings.embed_query("Delhi")
vector2 = embeddings.embed_query("Chocolate")


print(vector)
print(vector1)
print(vector2)