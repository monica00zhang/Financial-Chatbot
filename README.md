# Financial-Chatbot

# 🧾 Financial Chatbot with RAG + Streamlit

> A financial chatbot powered by OpenAI and Streamlit to analyze financial statements across multiple domains using RAG, T5 fine-tuning, and Buffett-style reasoning.

---

## 📌 Features

- ✅ Chatbot for interpreting financial statements across domains
- 🔎 RAG-based retrieval with chunk labeling + event extraction
- 📊 Streamlit UI with multi-sheet input: Income, Balance, Cash Flow
- 🧠 Embedding-enhanced retrieval from Buffett's past investment logic

---

## 🚀 Demo

👇 Click to try it out:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-demo-link.com)

---

## 🏗️ Project Structure

```bash
.
├── data/                       # Raw and preprocessed financial sheets
├── retriever/                 # RAG chunking, labeling, FAISS indexing
├── openai_api/                # OpenAI interface and generation logic
├── streamlit_app/             # Streamlit-based chatbot interface
├── run_financial_chatbot.py   # Main entrypoint script
└── README.md

```


## 🧠 Usage

```bash
streamlit run run_financial_chatbot.py
```

## 📈 Example Outputs
👇 Below is a sample demo of the chatbot answering questions based on uploaded financial reports:

![demo](assets/0513.gif)

