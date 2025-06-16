import re
import json

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

    keywords = {
        "leave policy": ["leave", "vacation", "time off"],
        "benefits": ["benefit", "insurance", "retirement"],
        "onboarding": ["start", "onboard", "new hire"]
    }

    for topic, keys in keywords.items():
        if any(k in text for k in keys):
            return topic, knowledge[topic]

    return None, None
