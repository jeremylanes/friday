# ⚡ Friday — Your Marvel AI Assistant

[![Python](https://img.shields.io/badge/python-3.13-blue?logo=python)](https://www.python.org/) 
[![LangChain](https://img.shields.io/badge/langchain-0.4.0-orange)](https://python.langchain.com/)
[![Cohere](https://img.shields.io/badge/Cohere-embed-multilingual-v3.0-purple)](https://cohere.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM-red)](https://console.groq.com/)

Friday is a **Marvel-powered AI assistant**, inspired by Tony Stark’s personal AI.  
Ask questions about the MCU, comics, and characters — Friday will give accurate, context-aware answers, with memory and retrieval from a local knowledge base.

---

## 💡 Features

- 🔹 **Contextual conversation** with memory buffer.  
- 🔹 **Local knowledge base** using FAISS + Cohere embeddings.  
- 🔹 **Generative responses** using Groq LLM (LLaMA3).  
- 🔹 Fully **extensible**: add more facts or sources.  
- 🔹 Lightweight console chat for fast experimentation.

---

## 📂 Project Structure

```yaml
friday/
├── data/ # Marvel facts (MCU + Comics)
│ ├── mcu_facts.txt
│ └── comics_facts.txt
├── db/ # FAISS index (generated locally)
├── build_index.py # Build the vector DB
├── chat.py # Console chatbot
├── .env # API keys (Cohere + Groq)
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

1. Clone the repository:

```bash
    git clone git@github.com:jeremylanes/friday.git
    cd friday
```
2. Create a Python virtual environment:

```bash
    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows
```
3. Install dependencies:

```bash
  pip install -r requirements.txt
```
4. Create a .env file at the root/src/:

```env
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
```
- 🔹 Get **Cohere** key: [Cohere Dashboard (free tier)](https://dashboard.cohere.com/api-keys)
- 🔹 Get **Groq** key: [Groq Console](https://console.groq.com/)

## 🗃️ Building the Knowledge Base (FAISS Index)
Before chatting, generate the embeddings database:

```bash
  python src/build_index.py
```
- 🔹 Files in ```data/``` are split into chunks and vectorized.
- 🔹 Index will be stored in ```db/marvel_index/```.
- 🔹 ```db/``` is gitignored, so everyone builds locally.

## 🤖 Running Friday
Start the console chatbot:

```bash
  python src/chat.py
```
- 🔹 Type your questions about Marvel.
- 🔹 Type exit or quit to leave.

### Example session:

```psql
🧑‍💬 You: Who is Thanos?
🤖 Friday: Thanos is a Titan obsessed with balancing the universe by destroying half of all life.
🧑‍💬 You: How does he die?
🤖 Friday: He is killed by Iron Man in Avengers: Endgame when Tony uses the Infinity Stones.
```
## ⚙️ Configuration
- 🔹 k in the retriever → number of passages retrieved for context.
- 🔹 temperature of LLM → controls creativity and verbosity.
- 🔹 Add more facts to data/*.txt → rebuild with build_index.py.

## 📌 Best Practices
- 🔹 .env never commit.
- 🔹 db/ is local, automatically ignored by git.
- 🔹 Always rebuild the index if data/ changes.
- 🔹 Keep chunks small (≈500 chars) for optimal retrieval.

---

## 🌟 Advanced Features (Optional)
- 🔹 Add web interface with FastAPI or HTMX.
- 🔹 Multi-index support (MCU, Comics, Series).
- 🔹 Combine local knowledge + web search for hybrid retrieval.
- 🔹 Experiment with Groq LLM settings for different response styles.

## 📝 .gitignore
```bash
.env
*.idea
db/
```

## 🏁 Goal
Learn the full LangChain workflow:

Loader → Splitter → Embeddings → VectorStore → Retriever → LLM → Memory

…and have a robust, extendable Marvel chatbot ready to deploy or integrate in any project.

---

## 🤝 Contributing
- 🔹 Add new facts in data/.
- 🔹 Rebuild the index with python build_index.py.
- 🔹 Test responses in chat.py.
- 🔹 Optional: improve prompts or add retrieval logic.