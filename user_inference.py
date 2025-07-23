import ray
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import socket
import numpy as np

# Start or connect to Ray
ray.init(address="auto")

print("Cluster resources:", ray.cluster_resources())

# Model name
model_name = "textattack/distilbert-base-uncased-imdb"

# Get only worker nodes (exclude head)
worker_nodes = [
    node for node in ray.nodes() 
    if node["Alive"] and not node["Resources"].get("node:" + ray.util.get_node_ip_address(), False)
]

if not worker_nodes:
    print("⚠️ No worker nodes available. Inference will happen on the head node.")
else:
    print("Available worker nodes:")
    for w in worker_nodes:
        print(" -", w["NodeManagerAddress"])

@ray.remote(num_cpus=1)
def run_inference(text_batch, model_name):
    hostname = socket.gethostname()
    print(f"Processing batch on {hostname}")
    
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

# --- Accept user input ---
print("\n📝 Enter up to 50 sentences for sentiment classification. Type 'done' to finish input.")
user_texts = []

while len(user_texts) < 50:
    text = input(f"Sentence {len(user_texts) + 1}: ")
    if text.strip().lower() == "done":
        break
    if text.strip():
        user_texts.append(text.strip())

if not user_texts:
    print("⚠️ No input text provided.")
    ray.shutdown()
    exit()

# --- Split into batches ---
batch_size = 5
text_batches = [user_texts[i:i + batch_size] for i in range(0, len(user_texts), batch_size)]

# --- Send to workers only ---
# Launch inference remotely
futures = [run_inference.options(
    scheduling_strategy="SPREAD"  # Distribute across nodes
).remote(batch, model_name) for batch in text_batches]

# Collect results
results = ray.get(futures)
predictions = [p for batch_preds in results for p in batch_preds]

# --- Display predictions ---
print("\n🔍 Predictions:")
for i, (text, pred) in enumerate(zip(user_texts, predictions)):
    label = "Positive" if pred == 1 else "Negative"
    print(f"{i+1:>2}. \"{text[:50]}...\" → {label}")

ray.shutdown()
