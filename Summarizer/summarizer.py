from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dotenv import load_dotenv
from State import BaseMessages
import av
from PIL import Image

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./Models/models--HuggingFaceTB--SmolVLM2-256M-Video-Instruct/067788b187b95ebe7b2e040b3e4299e342e5b8fd"
video_path = "./input/s2.mp4"

def summarizer_node(state: BaseMessages, model_path=model_path) -> BaseMessages:
    print("Entering the summarizer node")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model.to(device)
    video_path = state['video_path']
    sample_video = av.open(video_path)
    final_response = []

    question = "Summarize the video in detail. Also mention the time stamps for all the things mentioned."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": question}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    frame_data = generated_texts[0].split("Assistant: ")[-1].strip()
    print("The video summary is as follows:", frame_data)
    final_response.append(frame_data)
    state["video_context"].append(final_response)
    state["need_summarizer"] = 0 
    state['need_events'] = 1
    return state

