# Video Chat Assistant

A **multimodal conversational AI assistant** that lets users upload video files and engage in multi-turn conversations about video content. The system analyzes, summarizes, and supports both event-based and general discussions using advanced language and visual understanding modules.

## Project Overview

- **Upload videos** (MP4, MOV, AVI, MKV).
- **Get instant video summaries** using AI vision models.
- **Chat conversationally about video scenes, objects, and events** with memory across turns.
- **Modern chat UI** for seamless multimodal interaction.

Demo Video: https://drive.google.com/file/d/15fNnrsgh3f2WLBPPPkJ3B2oVD0sQBLQ4/view?usp=sharing

## Architecture Diagram

```

+----------------------+
| User Interface | (Electron/Browser: Drag & drop upload, chat UI)
+----------------------+
|
V
+----------------------+
| Flask REST API |
| - /upload_file |
| - /chat |
+----------------------+
|
V
+----------------------+
| LangGraph State |
| Workflow |
| (Entry, Summarizer, |
| Event Detector, |
| Multiturn Agent) |
+----------------------+
|
V
+----------------------+
| AI Modules: |
| - Video Summarizer |
| (e.g. SmolVLM2-256M)|
| - Chat Agent |
| (e.g., Llama 3.2-3b)|
+----------------------+

```

## Tech Stack Justification

### Backend

- **Flask**: Lightweight, ideal for RESTful APIs bridging frontend and AI logic.
- **LangGraph**: Orchestrates multistage workflows, supporting persistence and multimodal logic.

### AI Models

- **SmolVLM2 (or similar):** Efficient, fast, and suited for frame/sample video analysis.
- **Meta Llama or similar LLMs:** Provides natural, context-rich conversational responses.

### Frontend

- **HTML/CSS/JavaScript:** Modern UX, clarity, and usability for desktop/web via browser or Electron.

## Setup & Installation Instructions

### Requirements

- Python 3.10+
- Node.js (for optional Electron desktop packaging)
- pip

### 1. Clone the Repository

git clone https://github.com/DasPua/video-chat-assistant.git
cd video-chat-assistant


### 2. Set Up Python Environment

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


Your `requirements.txt` should include:

langchain
langgraph
langchain-community
huggingface_hub
opencv-python
transformers
torch
pillow
Torchvision
num2words
av
langchain_huggingface
grandalf
ultralytics
clip
Flask


**Additional Step:**

pip install https://github.com/openai/CLIP/archive/refs/heads/main.zip

Authenticate your HuggingFace Token:

1.In your terminal (Anaconda prompt, VSCode terminal, etc.), simply run:

``` huggingface-cli login ```

Paste your token when prompted. This saves it for all your local Python/HF code

Paste the huggingface token in the .env file

Download the Models from : https://drive.google.com/drive/folders/1VOfVzRaoaNKw1rYq6nvuWkskQ8vcZ38p?usp=sharing


### 3. Directory Structure

```
MANTRAHACKATHON/
├── venv/
├── Event_Detector/
│ └── event_detector.py
├── Models/
│ └── models--HuggingFaceTB--SmolVLM2-256M-Video-Instruct/
├── Multiturn_Agent/
│ └── agent.py
├── results/
├── State/
│ └── state.py
├── static/
│ └── input/
├── Summarizer/
│ └── summarizer.py
├── templates/
│ ├── frontend.html
│ └── index.html
├── .env
├── .gitattributes
├── .gitignore
├── app.py
├── requirements.txt
├── testing.py
├── yolov8s-world.pt
└── yolov8s.pt

```

text


### 4. Model Downloads

- Download or place your vision & language model folders (e.g., SmolVLM2, Llama) inside `Models/` as referenced in your code.

### 5. Running the Application

Start Flask backend:

python app.py


By default, this runs at [http://localhost:5000/](http://localhost:5000/).

## Usage Instructions

### Step 1: Access the UI

- Open your browser: `http://localhost:5000/chat_ui`

### Step 2: Upload a Video

- Click the "+" button in the chat input strip.
- Select and upload a video file (MP4, AVI, MOV, MKV supported).

### Step 3: Chat About the Video

- Enter a question or prompt (e.g., “What happens in this video?” or “Describe the important events”).
- Click send (→) or press Enter.
- Instantly receive rich, AI-generated answers grounded in your video.

## Example Interactions

- **Upload:** `example_clip.mp4`
    - **Prompt:** "Summarize the main activity in the first minute."
    - **Response:** "The video shows people entering a conference room, greeting each other and preparing for a meeting."
    - **Prompt:** "Is there any unusual event after 20 seconds?"
    - **Response:** "At around 27 seconds, the lights flicker briefly and participants appear surprised."

## Advanced Features

- **Efficient Processing:** Video content is processed once per session.
- **Contextual Memory:** Multi-turn chat remembers prior context and queries within a session.
- **Decoupled Endpoints:** File upload and chat are separated for clean UX and easy extensibility.

**You now have a production-ready, multimodal chat assistant for video understanding!**


