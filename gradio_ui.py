import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import re


def load_knowledge(path="custom_knowledge.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_input(text):
    text = text.lower().strip()
    return re.sub(r'[^\w\s]', '', text)

def match_hr_topic(text, knowledge):
    text = preprocess_input(text)
    for topic, response in knowledge.items():
        if topic in text:
            return topic, response
    return None, None

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

knowledge_base = load_knowledge()
chat_history = []
context_stack = []

def chat_with_bot(user_input):
    topic, response = match_hr_topic(user_input, knowledge_base)
    if response:
        chat_history.append(user_input)
        chat_history.append(response)
        context_stack.append(topic)
        return response

    context = f"[HR topic: {context_stack[-1]}] " if context_stack else ""
    full_input = context + " ".join(chat_history[-4:] + [user_input])
    output = pipe(full_input, max_length=100, do_sample=True, top_p=0.9)[0]['generated_text']
    chat_history.append(user_input)
    chat_history.append(output)
    return output


gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(lines=2, placeholder="Ask an HR-related question..."),
    outputs="text",
    title="🤖 HR Assistant Bot",
    description="This assistant uses both rule-based answers and AI-generated replies for HR support.",
    theme="default"
).launch()
