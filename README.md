This repository contains two Python scripts that demonstrate distributed sentiment classification using Ray, Hugging Face Transformers, and the textattack/distilbert-base-uncased-imdb model:

imdb_inference.py: Runs sentiment inference on a subset of the IMDb test dataset in parallel.

user_inference.py: Accepts custom user input and performs distributed inference interactively.

🧠 Model Used
textattack/distilbert-base-uncased-imdb: A fine-tuned version of DistilBERT trained on the IMDb sentiment classification task (binary: positive/negative).

🚀 Dependencies
Install the required dependencies with:

pip install torch transformers datasets ray
Ensure that a Ray cluster is running. You can start a local cluster for testing:
ray start --head
Or connect to an existing one with worker nodes.



⚙️ Features
Ray Remote Functions: Each worker loads its own copy of the model and tokenizer.

Node Awareness: Detects available Ray nodes and uses SPREAD strategy to balance the load.

Scalability: Easily scales inference across CPU workers in a cluster.

📊 Sample Output
📝 Enter up to 50 sentences for sentiment classification. Type 'done' to finish input.
Sentence 1: The movie was absolutely fantastic!
Sentence 2: I didn't enjoy it at all.
done

🔍 Predictions:
 1. "The movie was absolutely fantastic!..." → Positive
 2. "I didn't enjoy it at all...." → Negative
🧹 Cleanup
Stop the Ray cluster if you're done:
ray stop
📌 Notes
Designed to work with CPU-based Ray workers (no GPU usage).

Each task allocates 1 CPU (@ray.remote(num_cpus=1)).

Use scheduling_strategy="SPREAD" to maximize node utilization.

📄 License
This project is for educational purposes and follows the licenses of the associated libraries and models.
