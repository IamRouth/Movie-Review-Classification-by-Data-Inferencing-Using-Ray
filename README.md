This repository contains two Python scripts that demonstrate distributed sentiment classification using Ray, Hugging Face Transformers, and the textattack/distilbert-base-uncased-imdb model:

imdb_inference.py: Runs sentiment inference on a subset of the IMDb test dataset in parallel.

user_inference.py: Accepts custom user input and performs distributed inference interactively.

🧠 Model Used
textattack/distilbert-base-uncased-imdb: A fine-tuned version of DistilBERT trained on the IMDb sentiment classification task (binary: positive/negative).

🚀 Dependencies
Install the required dependencies with:

bash
Copy
Edit
pip install torch transformers datasets ray
Ensure that a Ray cluster is running. You can start a local cluster for testing:

bash
Copy
Edit
ray start --head
Or connect to an existing one with worker nodes.

📂 File Overview
1. imdb_inference.py
This script:

Connects to a Ray cluster

Loads a small subset of the IMDb test set (first 150 reviews)

Distributes inference across all available Ray nodes

Prints the first 10 sentiment predictions

Usage
bash
Copy
Edit
python imdb_inference.py
2. user_inference.py
This script:

Connects to a Ray cluster

Accepts up to 50 custom user sentences via command-line

Splits input into batches and distributes inference across available nodes

Prints sentiment labels for each input

Usage
bash
Copy
Edit
python user_inference.py
Type done to end input early.

⚙️ Features
Ray Remote Functions: Each worker loads its own copy of the model and tokenizer.

Node Awareness: Detects available Ray nodes and uses SPREAD strategy to balance the load.

Scalability: Easily scales inference across CPU workers in a cluster.

📊 Sample Output
vbnet
Copy
Edit
📝 Enter up to 50 sentences for sentiment classification. Type 'done' to finish input.
Sentence 1: The movie was absolutely fantastic!
Sentence 2: I didn't enjoy it at all.
done

🔍 Predictions:
 1. "The movie was absolutely fantastic!..." → Positive
 2. "I didn't enjoy it at all...." → Negative
🧹 Cleanup
Stop the Ray cluster if you're done:

bash
Copy
Edit
ray stop
📌 Notes
Designed to work with CPU-based Ray workers (no GPU usage).

Each task allocates 1 CPU (@ray.remote(num_cpus=1)).

Use scheduling_strategy="SPREAD" to maximize node utilization.

📄 License
This project is for educational purposes and follows the licenses of the associated libraries and models.
