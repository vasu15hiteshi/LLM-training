🧠 Personalized LLM Trained on WhatsApp Chats

Fine-tuned transformer-based language model using personal chat data to emulate individual conversational style.

📌 Project Overview

This project involves building a custom Large Language Model (LLM) that mimics my personal communication style by training it on 51k tokens extracted from my WhatsApp chats. It leverages transformer-based architectures, byte pair encoding (BPE) tokenization, LangChain for dynamic context management, and instruction + LoRA fine-tuning techniques.

⸻

🔧 Tech Stack
	•	Python
	•	LangChain
	•	Transformers (HuggingFace)
	•	LoRA (Low-Rank Adaptation)
	•	BPE Tokenization
	•	Jupyter Notebooks

⸻

📊 Key Features
	•	Tokenizes personal WhatsApp chat history using the Byte Pair Encoding (BPE) algorithm.
	•	Pre-trains a base transformer model on sequential dialogue data.
	•	Creates a curated instruction-tuning dataset for supervised fine-tuning.
	•	Applies LoRA to reduce computational cost while achieving high personalization.
	•	Integrates with LangChain to maintain conversation memory and contextual awareness.

⸻

🧪 Workflow
	1.	Data Extraction: Parsed and cleaned WhatsApp chat data to form training corpus.
	2.	Tokenization: Applied BPE to encode text into efficient token format.
	3.	Pre-Training: Initialized transformer model on generic text patterns.
	4.	Fine-Tuning:
	•	Instruction tuning using custom prompts
	•	LoRA applied for lightweight fine-tuning
	5.	Deployment (WIP): Planning LangChain agent integration for interactive chat via CLI or web.

🚀 Future Improvements
	•	Add a web interface using Flask or FastAPI.
	•	Evaluate response accuracy using BLEU/ROUGE metrics.
	•	Integrate real-time inference API for demo purposes.

⸻

🧠 Inspiration

Inspired by the desire to personalize AI assistants and explore lightweight, domain-specific LLM training using real-world private datasets.

⸻

📜 License

This project is for educational and personal research use only. All training data is anonymized and not distributed.
