from State import BaseMessages
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()

model_repo = "meta-llama/Llama-3.2-3B-Instruct"

def agent_node(state: BaseMessages, model_repo=model_repo) -> BaseMessages:
    print("entering the agent node")
    llm = HuggingFaceEndpoint(
        repo_id=model_repo,
        task="text-generation",
        temperature=0.5,
    )
    llm = ChatHuggingFace(llm=llm)

    messages = [
    SystemMessage(
        content=(
            "You are a helpful and intelligent assistant. Use all the information provided to answer the user's query accurately and thoroughly.\n\n"
            "Here is the context to assist you:\n"
            f"- **Chat History**: {state['history']}\n"
            f"- **Video Summary**: {state['video_context']}\n"
            f"- **Relevant Events**: {state['events']}\n\n"
           f"If the user asks about specific events, refer to the event information above to generate your response. Be concise, clear, and informative in your answers."
        )
    ),
    HumanMessagePromptTemplate.from_template("{user_query}")
]

    prompt = ChatPromptTemplate.from_messages(messages)
    responder_chain = prompt | llm

    last_user_query = state["history"][-1].content
    response_obj = responder_chain.invoke({"user_query": last_user_query})
    text = response_obj.content if hasattr(response_obj, "content") else str(response_obj)

    state["response"].append(text)
    state["history"].append(AIMessage(content=text))
    return state

# state  = {
#     "messages" : ["tell me who are you?"],
#     "response" : []
# }
# response = agent_node(state)
# print(response)