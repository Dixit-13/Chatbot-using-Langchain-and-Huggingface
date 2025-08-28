from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='google/flan-t5-large',
    task='summarization',
    pipeline_kwargs=dict(
        max_length=50,
        min_length=10,
        temperature=0.7
    )
)
model = llm

messages = [
    SystemMessage(content="You are a helpful assistant that summarizes research papers."),
    HumanMessage(content="Tell me about LangChain.")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result))

print(messages)