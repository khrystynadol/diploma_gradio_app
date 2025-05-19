import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CustomClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.3):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.base.gradient_checkpointing_enable()
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return {'logits': logits}


MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
NUM_LABELS = 7
MODEL_PATH = "models/best_full_go_emotions_model.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = CustomClassifier(MODEL_NAME, num_labels=NUM_LABELS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()


def classify_text_chunks(text_chunks):
    encodings = tokenizer(text_chunks, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"]
        )
    logits = outputs["logits"]
    preds = torch.argmax(logits, dim=1)
    return preds.numpy()
