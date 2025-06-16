from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import re

app = Flask(__name__)

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

# load model
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)


hr_knowledge = load_knowledge()
chat_history = []
hr_context = []

def ask_bot(user_input):
    topic, hr_response = match_hr_topic(user_input, hr_knowledge)
    if hr_response:
        chat_history.append(user_input)
        chat_history.append(hr_response)
        hr_context.append(topic)
        return hr_response

    context = f"[HR topic: {hr_context[-1]}] " if hr_context else ""
    full_input = context + " ".join(chat_history[-4:] + [user_input])
    output = pipe(full_input, max_length=100, do_sample=True, top_p=0.9)[0]['generated_text']
    chat_history.append(user_input)
    chat_history.append(output)
    return output

# end point api 
@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    response = ask_bot(user_message)
    return jsonify({"response": response})

# run server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
