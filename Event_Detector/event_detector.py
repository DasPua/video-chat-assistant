import cv2
import torch
import numpy as np
import os
import math
from PIL import Image
from typing import Dict
import warnings
from State import BaseMessages
warnings.filterwarnings("ignore")
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥 Using device: {device.upper()}")


# === YOLO-WORLD ===
class YOLOWorldDetector:
    def __init__(self, model_size='s'):
        from ultralytics import YOLO
        model_name = f'yolov8{model_size}-world.pt'
        self.model = YOLO(model_name)
        self.model.to(device)
        print(f"✅ YOLO-World {model_size.upper()} model loaded on {device}.")

    def detect_objects(self, frame, classes, confidence=0.3):
        self.model.set_classes(classes)
        results = self.model(frame, conf=confidence, verbose=False)
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id < len(classes):
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': classes[cls_id],
                            'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2),
                            'source': 'YOLO'
                        })
        return detections


# === OWL-ViT ===
class OWLViTDetector:
    def __init__(self):
        from transformers import pipeline
        self.detector = pipeline(
            model="google/owlvit-base-patch32",
            task="zero-shot-object-detection",
            device=0 if device == "cuda" else -1
        )
        print("✅ OWL-ViT model loaded on", device)

    def detect_objects(self, frame, classes, confidence=0.3):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        candidate_labels = [f"a photo of a {cls}" for cls in classes]
        predictions = self.detector(image, candidate_labels=candidate_labels, threshold=confidence)
        detections = []
        for pred in predictions:
            box = pred['box']
            label = pred['label'].replace("a photo of a ", "")
            detections.append({
                'bbox': [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
                'confidence': pred['score'],
                'class': label,
                'center': ((box['xmin'] + box['xmax']) // 2, (box['ymin'] + box['ymax']) // 2),
                'source': 'OWL'
            })
        return detections


# === UNIVERSAL DETECTOR ===
class UniversalObjectDetector:
    def __init__(self, model_size='s', use_owl=True):
        self.yolo = YOLOWorldDetector(model_size)
        self.use_owl = use_owl
        self.owl = OWLViTDetector() if use_owl else None
        self.events = []

    def parse_input(self, user_input: str) -> Dict[str, str]:
        result = {}
        phrases = user_input.lower().split(',')
        for phrase in phrases:
            phrase = phrase.strip()
            words = phrase.split()
            if len(words) == 1:
                result[words[0]] = None
            elif len(words) >= 2:
                action = words[-1]
                obj = ' '.join(words[:-1])
                result[obj] = action
        return result

    def detect_motion(self, detections, prev_positions, obj_class, timestamp):
        results = []
        for det in detections:
            if det['class'] != obj_class:
                continue
            center = det['center']
            action = None
            if obj_class in prev_positions:
                prev_center = prev_positions[obj_class]
                movement = math.dist(center, prev_center)
                if movement > 30:
                    action = "moving"
                elif movement > 15:
                    action = "walking"
                elif movement < 5:
                    action = "still"
            prev_positions[obj_class] = center
            det['action'] = action
            det['timestamp'] = timestamp
            results.append(det)
        return results

    def process_video(self, video_path, object_action_dict: Dict[str, str], output_dir="results"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = int(fps)  # 1 fps
        os.makedirs(output_dir, exist_ok=True)

        final_events = []

        for obj, expected_action in object_action_dict.items():
            print(f"\n🔍 Detecting: {obj} with action: {expected_action or 'any'}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0
            prev_positions = {}
            self.events = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if count % interval == 0:
                    ts = count / fps

                    yolo_dets = self.yolo.detect_objects(frame, [obj])
                    owl_dets = self.owl.detect_objects(frame, [obj]) if self.use_owl else []

                    combined_dets = yolo_dets + [
                        d for d in owl_dets if d['class'] != "" and d['class'] not in [yd['class'] for yd in yolo_dets]
                    ]

                    dets = self.detect_motion(combined_dets, prev_positions, obj, ts)

                    if expected_action:
                        dets = [d for d in dets if d['action'] == expected_action]

                    for d in dets:
                        box = d['bbox']
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        label = f"{d['class']} {d['confidence']:.2f} [{d['source']}]"
                        if d.get('action'):
                            label += f" ({d['action']})"
                        cv2.putText(frame, label, (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.events.append(d)
                        cv2.imwrite(f"{output_dir}/{obj}_{count:06d}.jpg", frame)

                count += 1

            print(f"📋 SUMMARY for '{obj}' ({expected_action or 'any'}): {len(self.events)} detections")
            for e in self.events:
                print(f"[{e['timestamp']:.1f}s] {e['class']} ({e['confidence']:.2f}) - {e.get('action', 'N/A')} [{e['source']}]")
            final_events.extend(self.events)

        cap.release()
        return final_events

model_repo = "meta-llama/Llama-3.2-3B-Instruct"

def detection_node(state: BaseMessages, model_repo = model_repo)->BaseMessages:
    print("entering the detection node")
    llm = HuggingFaceEndpoint(
        repo_id=model_repo,
        task="text-generation",
        temperature=0.5,
    )
    llm = ChatHuggingFace(llm=llm)

    messages = [
    SystemMessage(
        content=("""you are an intelligent agent that identifies objects and action from user's input.
                 You will be given a sentence and you have to identify the objects and actions and return the ouput in comma separated values. Here is an example given below:
                 Example : 
                 input : can you find the when the aeroplane is flying and the dog is running ?
                 output : aeroplane flying, dog running
                 
                 If there is no action for a object then only return the object.
                 Example : 
                 input : is there a truck and cat in the video?
                 output : truck,cat
                 Only return the output in the above mentioned format and do not return anything else and no special words like None.""")
    ),
    HumanMessagePromptTemplate.from_template("{user_query}")
]

    prompt = ChatPromptTemplate.from_messages(messages)
    responder_chain = prompt | llm

    last_user_query = state["history"][-1].content
    response_obj = responder_chain.invoke({"user_query": last_user_query})
    text = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
    print("checking : ", text)
    user_cmd = text
    vid_path  = state["video_path"]
    detector  = UniversalObjectDetector(use_owl=True)
    plan      = detector.parse_input(user_cmd)
    events    = detector.process_video(vid_path, plan)
    state["events"].append(events)
    print("the events that have occurred so far are:", state['events'])
    return state