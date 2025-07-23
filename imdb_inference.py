import ray
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import torch
import socket
import numpy as np

# Initialize Ray (connect to existing cluster)
ray.init(address="auto")

# Verify cluster resources
print("Cluster resources:", ray.cluster_resources())

# Load dataset
dataset = load_dataset("imdb", split="test").select(range(150))
model_name = "textattack/distilbert-base-uncased-imdb"

@ray.remote(num_cpus=1)  # Specify CPU requirements here
def run_inference(text_batch, model_name):
    """Runs inference on a batch of texts and returns predictions."""
    hostname = socket.gethostname()
    print(f"Processing batch on {hostname}")
    
    # Load model (each worker loads its own copy)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name).to("cpu")
    
    inputs = tokenizer(
        text_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to("cpu")

    with torch.no_grad():
        outputs = model(**inputs)
    
    return torch.argmax(outputs.logits, dim=-1).tolist()

# Batch processing
batch_size = 10
text_batches = [dataset[i:i+batch_size]["text"] for i in range(0, len(dataset), batch_size)]

# Get all available nodes
nodes = list(ray.nodes())
print(f"Available nodes: {[node['NodeManagerAddress'] for node in nodes]}")

# Distribute batches
futures = [run_inference.remote(batch, model_name) for batch in text_batches]

# Collect results
results = ray.get(futures)
predictions = [pred for batch_preds in results for pred in batch_preds]

# Print first 10 predictions
for i in range(10):
    print(f"Text: {dataset[i+20]['text'][:10]}... | Prediction: {'Positive' if predictions[i] == 1 else 'Negative'}")

ray.shutdown()
