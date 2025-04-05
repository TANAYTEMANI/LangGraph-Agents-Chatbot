# LangGraph-Agents-Chatbot
## 🌤️📄 Weather & PDF Query Chatbot using LangGraph + LangChain

A LangGraph-based agentic pipeline chatbot that supports **real-time weather information retrieval** via OpenWeatherMap API and **question answering from PDF documents** using **RAG (Retrieval-Augmented Generation)**.

This project demonstrates how to combine multiple tools — LangChain, LangGraph, Qdrant, OpenWeatherMap, and LangSmith — to create an intelligent, multi-functional chatbot with a user-friendly Streamlit UI.

---

## 🔧 Features

- 🌦️ **Real-Time Weather Fetching** using [OpenWeatherMap API](https://openweathermap.org/api)
- 📄 **PDF Question Answering** using **RAG** (embeddings + vector database)
- 🧠 **LangGraph Decision Node** to dynamically route between functionalities
- 🧬 **Embeddings Generation** with LangChain and storage in **FAISS**
- 🔍 **Semantic Search** and **Answer Summarization** using an LLM
- 🧪 **Unit Testing** for APIs, LLM logic, and retrieval system
- 🧪 **LangSmith Evaluation** for LLM responses
- 💬 **Streamlit UI** with a chatbot interface for easy user interaction

---

## 🧰 Tech Stack

| Tech           | Description                                      |
|----------------|--------------------------------------------------|
| LangGraph      | Agentic pipeline and conditional node execution  |
| LangChain      | LLM pipeline, RAG, and tool integrations         |
| FAISS          | Vector database for storing document embeddings  |
| OpenWeatherMap | API to fetch current weather data                |
| Streamlit      | UI framework for chatbot interface               |
| LangSmith      | Evaluation of LLM output quality                 |
| Pytest         | For test coverage of logic and API integrations  |


## 🚀 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/langgraph-weather-pdf-chatbot.git
cd langgraph-weather-pdf-chatbot
```

2. **Create & Activate Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Application**

```bash
streamlit run app.py
```
