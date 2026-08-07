from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model



model = init_chat_model(
    "mistralai:mistral-small-2603"
)

paragraph = input("Enter the paragraph: ")

prompt = f'''From the given paragraph Extract the following details and return the response in JSON format.
Movie 
Title
Genre
Release Year
Director
Lead Cast
Music Composer
Core Plot / Premise
Key Settings & Themes
Audience / Critical Reception
Review Ratings

Here is the paragraph:
{paragraph}
'''

response = model.invoke(prompt)

print(response.content)