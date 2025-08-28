from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # 👈 real chat model
    task="text-generation",
    huggingfacehub_api_token="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",
    MODEL_kwargs=dict(
        max_length=1000,
        temperature=0.7,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input('You:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content) 
    
    
print(chat_history) # Assuming the response has a 'content' attribute




# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # 👈 real chat model
#     task="text-generation",
#     huggingfacehub_api_token="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"
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India?")
# print(result.content)



