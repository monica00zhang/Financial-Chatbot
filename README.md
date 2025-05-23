# Financial-Chatbot

> A financial chatbot fine-tuned on T5, enhanced with RAG and OpenAI API to analyze and interpret financial statements through Buffett-style reasoning.

---

## 📌 Key Features

1. **Financial Statement Visualization**:  
   Select and visualize key metrics (e.g. Gross Margin) from income, balance, and cash flow sheets over multiple years or a specific year.

2. **Conversational Analysis**:  
   Chat with the model to understand what specific financial metrics mean for a company. The model replies in the tone and principles of Warren Buffett’s investment philosophy.

3. **Contextual Interpretation**:  
   For example, asking _“What does a continuously decreasing gross margin mean for a company like XX?”_ will trigger an explanation covering operational implications.

4. **Buffett-style Explanations on Demand**:  
   Click the **“Explain in Buffett’s Style”** button to generate insights based on historical shareholder letters and value-investment heuristics.

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

1. 📊 Visualization of financial statement trends over the past 3 years.
![demo](assets/vis_sheets.gif)
   
2. 📅 Visualization of financial sheets for a specific year.
![demo](assets/vis_sheets2.gif)
   
3. 💬 Chat-based interpretation of a financial metric’s long-term trend.
4. 🪙 Buffett-style investment commentary on company fundamentals.
![demo](assets/chat_func1.gif)

5. 🧠 Chat-based explanation of a financial metric for a specific year (e.g., 2024).
![demo](assets/chat_func2.gif)




![demo](assets/0513.gif)

