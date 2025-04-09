from transformers import pipeline

text_classifier = pipeline("sentiment-analysis")

def analyze_text_sentiment(text: str):
    result = text_classifier(text)[0]
    return result["label"], result["score"]