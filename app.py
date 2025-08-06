from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dotenv import load_dotenv
from State import BaseMessages
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from Multiturn_Agent import agent_node
from Summarizer import summarizer_node
from Event_Detector import detection_node
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

# Load environment variables
env_loaded = load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Setup LangGraph Pipeline ---------
graph = StateGraph(state_schema=BaseMessages)
AGENT = "agent_node"
SUMMARIZER = "summarizer_node"
ENTRY = "entry"
DETECTION = "detection"

def entry(state: BaseMessages) -> BaseMessages:
    print("entering the entry node")
    # Only trigger summarizer once per video
    if state.get("video_path") is not None and state.get("need_summarizer", 0) == 0:
        state['need_summarizer'] = 1
    return state


def entry_router(state: BaseMessages):
    print(f"Routing decision - need_summarizer: {state.get('need_summarizer')}, "
          f"video_context: {state.get('video_context')}, events: {state.get('events')}")
    if state.get('events'):
        return AGENT
    elif state.get('video_context'):
        return DETECTION
    elif state.get('need_summarizer'):
        return SUMMARIZER
    return AGENT

# Add nodes and edges
graph.add_node(AGENT, agent_node)
graph.add_node(SUMMARIZER, summarizer_node)
graph.add_node(ENTRY, entry)
graph.add_node(DETECTION, detection_node)
graph.set_entry_point(ENTRY)

graph.add_conditional_edges(
    ENTRY,
    entry_router,
    {AGENT: AGENT, DETECTION: DETECTION, SUMMARIZER: SUMMARIZER}
)
graph.add_edge(SUMMARIZER, AGENT)
graph.add_edge(DETECTION, AGENT)
graph.add_edge(AGENT, END)

# Checkpointer and compiled app
memory = MemorySaver()
main_app = graph.compile(checkpointer=memory)
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# --------- Flask Application ---------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = './static/input'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global shared state (single-user)
state = {
    "messages": [],
    "history": [],
    "video_path": None,
    "response": [],
    "video_context": [],
    "need_summarizer": 0,
    "events": [],
    "need_events": 0
}
current_video_path = None

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return jsonify(file_path=path)

@app.route('/chat', methods=['POST'])
def chat():
    global state, current_video_path

    data = request.json or {}
    query = data.get('query', '')
    fp = data.get('file_path')

    # Reset state only if a new video is uploaded
    if fp and fp != current_video_path:
        print("New video uploaded, resetting state.")
        current_video_path = fp
        state.update(
            video_path=fp,
            need_summarizer=0,
            need_events=0,
            video_context=[],
            events=[],
            response=[],
            messages=[],
            history=[]
        )

    # Append user inputs to state
    state['messages'].append(query)
    state['history'].append(HumanMessage(content=query))

    try:
        new_state = main_app.invoke(state, config=thread_config)
        # Persist updated state
        state.update(new_state)
        reply = new_state['response'][-1]
        return jsonify(response=reply)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
