from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM-135M-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("Who built Jantar Mantar in India? answer in one sentence.")
print(response.content)