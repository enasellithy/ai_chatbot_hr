from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils import load_knowledge, match_hr_topic

# Load model
mname = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(mname)
model = AutoModelForSeq2SeqLM.from_pretrained(mname)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Load HR knowledge
hr_knowledge = load_knowledge()
chat_history = []
hr_context = []

def ask_bot(user_input):
    topic, hr_response = match_hr_topic(user_input, hr_knowledge)
    if hr_response:
        hr_context.append(topic)
        chat_history.append(user_input)
        chat_history.append(hr_response)
        return hr_response.strip()

    context = f"[HR topic: {hr_context[-1]}] " if hr_context else ""
    full_input = context + " ".join(chat_history[-4:] + [user_input])
    output = pipe(full_input, max_length=100, do_sample=True, top_p=0.9)[0]['generated_text']
    chat_history.append(user_input)
    chat_history.append(output)
    return output.strip()

# CLI interaction
print("HR Assistant: Hello! I'm your HR assistant. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("HR Assistant: Goodbye!")
        break
    print("HR Assistant:", ask_bot(user_input))
