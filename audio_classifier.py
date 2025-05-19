import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

# audio_model_path = "models/ravdess-tess-hubert-model-7.pth"
audio_model_path = "models/mc-eiu-hubert-model.pth"
hubert_model = HubertForSequenceClassification.from_pretrained(
    "superb/hubert-large-superb-er",
    num_labels=7,
    ignore_mismatched_sizes=True
)
hubert_model.load_state_dict(torch.load(audio_model_path, map_location=torch.device('cpu')))
hubert_model.eval()

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")


def classify_audio_chunks(audio_chunks):
    print("Here:")
    inputs = []
    for chunk in audio_chunks:
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(chunk.array_type).max
        inputs.append(samples)

    print(f"Number of chunks: {len(inputs)}")
    print(f"Shape of first input array: {inputs[0].shape}")

    encodings = feature_extractor(
        inputs,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    print(f"Input values tensor shape: {encodings['input_values'].shape}")

    with torch.no_grad():
        outputs = hubert_model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions
