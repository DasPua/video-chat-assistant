from transformers.pipelines import pipeline
from State import BaseMessages
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()

model_repo = "meta-llama/Llama-3.2-3B-Instruct"

def agent_node(state : BaseMessages, model_repo = model_repo) ->BaseMessages :
    llm = HuggingFaceEndpoint(
        repo_id=model_repo,
        task = "text-generation",
        temperature=0.5,
    )

    llm = ChatHuggingFace(llm = llm)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="Respond to all user queries in a clear, concise, and accurate manner. Prioritize brevity and relevance. Avoid unnecessary details."
        ),
        HumanMessagePromptTemplate.from_template("{user_query}")
    ])
    responder_chain = prompt|llm
    last_user_query = state["messages"][-1]
    response = responder_chain.invoke({"user_query" : last_user_query})
    print(response)
    state['response'].append(response)
    return state

state  = {
    "messages" : ["tell me who are you?"],
    "response" : []
}
response = agent_node(state)
print(response)