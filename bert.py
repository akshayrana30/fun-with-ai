from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Sample text for classification
text = "I love machine learning and artificial intelligence!"

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)

print(f"Text: {text}")
print(f"Predicted class: {predicted_class.item()}")
print(f"Class probabilities: {predictions}")
