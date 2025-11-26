from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
# Note: This is a general pre-trained model. For real applications,
# the model should be fine-tuned for your specific classification task.
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
# Note: Class indices are not directly interpretable without fine-tuning
# This demonstrates the model structure and prediction process
print(f"Predicted class: {predicted_class.item()}")
print(f"Class probabilities: {predictions}")
