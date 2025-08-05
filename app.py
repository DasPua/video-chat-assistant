from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from State import BaseMessages
from Multiturn_Agent import agent_node
from Summarizer import summarizer_node
from langgraph.graph import END
import uuid
from Event_Detector import detection_node

graph = StateGraph(state_schema=BaseMessages)

AGENT = "agent_node"
SUMMARIZER = "summarizer_node"
ENTRY = "entry"
DETECTION = "detection"

def entry(state : BaseMessages)->BaseMessages:
    print("entering the entry node")
    if state["video_path"] is not None :
        state['need_summarizer']= state['need_summarizer']+1
    return state

def entry_router(state:BaseMessages):
    if len(state['events']):
        return AGENT
    if len(state['video_context']) : 
        return DETECTION
    if state['need_summarizer'] ==1:
        return SUMMARIZER
    return AGENT


graph.add_node(AGENT, agent_node)
graph.add_node(SUMMARIZER, summarizer_node)
graph.add_node(ENTRY, entry)
graph.add_node(DETECTION, detection_node)
graph.set_entry_point(ENTRY)

graph.add_conditional_edges(ENTRY,
    entry_router,
    {
        SUMMARIZER: SUMMARIZER,
        AGENT: AGENT,
        DETECTION:DETECTION
    }
)
graph.add_edge(SUMMARIZER,AGENT)
graph.add_edge(DETECTION, AGENT)
graph.add_edge(AGENT, END)


memory = MemorySaver()
main_app = graph.compile(checkpointer=memory)

thread_config = {"configurable": {
    "thread_id": uuid.uuid4()
}}

print(main_app.get_graph().draw_mermaid())
main_app.get_graph().print_ascii()

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

# ◀— all your LangGraph imports & graph.compile(...) go here

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = './static/input'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global initial state (you can refactor to per-session later)
state = {
    "messages": [], "history": [], "video_path": None,
    "response": [], "video_context": [],
    "need_summarizer": 0, "events": [], "need_events": 0
}

def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in {"mp4","mov","avi","mkv"}

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/chat_ui')
def chat_ui():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify(error="Invalid or missing file"), 400
    fn = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    file.save(path)
    return jsonify(file_path=path)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    query = data.get('query','')
    fp = data.get('file_path')
    if fp:
        state.update(video_path=fp,
                     need_summarizer=0,
                     video_context=[],
                     events=[])
    state["messages"].append(query)
    state["history"].append(HumanMessage(content=query))

    try:
        new_state = main_app.invoke(state, config=thread_config)
        print(new_state)
        reply = new_state["response"][-1]
        return jsonify(response=reply)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
