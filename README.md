# 🔍 Smart Image Search Engine 

An intelligent **AI-powered Image Search Engine** that uses **CLIP embeddings**, **Qdrant vector search**, and **LangChain agents** to retrieve visually and semantically similar images.  
It allows users to search either **by text** or **by uploading an image**, and automatically decides which search strategy to use using an **LLM Agent (Groq API)**.

---

## 🚀 Features

- **Multimodal Search:** Search images by **text descriptions** or **image queries**.  
- **Agent-based Automation:** An LLM-powered agent chooses the best search tool (feature, metadata, hybrid, or random).  
- **CLIP Embeddings:** Extracts visual and textual embeddings for efficient semantic similarity.  
- **Qdrant Integration:** High-performance vector database for similarity search.  
- **Metadata Search:** Fetches results using an external metadata API (Cumulus).  
- **Streamlit UI:** Interactive and modern web interface.  
- **Caching Optimized:** Efficient indexing and model loading with `@st.cache_resource`.

---

## 🧩 Project Structure

```
Smart-Image-Search/
│
├── app.py                          # Streamlit UI and main application
├── .env                            # Environment variables
├── requirements.txt                # All dependencies
│
├── config/
│   └── settings.py                 # Configuration and environment setup
│
├── agents/
│   ├── agent_executor.py           # Groq LLM agent initialization and execution
│   ├── prompts.py                  # System prompts for the agent
│   └── tools.py                    # Tool definitions for agent
│
├── services/
│   └── search_services.py          # Core search logic (image, text, metadata, hybrid)
│
├── endpoints/
│   └── api_endpoints.py            # API client for metadata search
│
├── utils/
│   ├── clip_helper.py              # CLIP model utilities (embedding generation)
│   ├── qdrant_helper.py            # Qdrant client operations
│   ├── helpers.py                  # Image loading and validation utilities
│   └── ui_helpers.py               # Streamlit result display helpers
│
└── image_store/                    # Local image storage directory
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/amit-maheshwari-viitorcloud/Smart-Image-Search-Engine.git
cd Smart-Image-Search-Engine
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**  
  ```bash
  venv\Scripts\activate
  ```
- **Linux / macOS:**  
  ```bash
  source venv/bin/activate
  ```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in your project root:

```bash
# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = "6333"
COLLECTION_NAME = "image_embeddings"
EMBEDDING_DIMENSION_SIZE = "512"

# API Configuration
GROQ_API_KEY = "Your-Groq-Api-Key"

# OAuth Configuration
CUMULUS_CLIENT_ID = "CUMULUS-IDCCG-PUB"
CUMULUS_API_KEY = "41204aed-89d7-4a45-9455-976ac475a8ab"

# Model Configuration
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_STORE_PATH = "image_store"
```

> ⚠️ Replace `Your-Groq-Api-Key` with your actual key from [Groq Console](https://console.groq.com).

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧠 How It Works

1. **Initialization:**  
   Loads environment variables and initializes CLIP, Qdrant, and Groq LLM agent and fetch data via API.

2. **User Input:**  
   Input text (`"flower"`) or upload an image.

3. **Agent Decision:**  
   The LLM agent decides the appropriate search:
   - `search_by_feature`
   - `search_by_metadata`
   - `search_hybrid`
   - `random_search`

4. **Result Display:**  
   Shows most relevant images to user.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python 3.10+ |
| Framework | Streamlit |
| LLM Backend | Groq API |
| Vector Database | Qdrant |
| Embedding Model | OpenAI CLIP (`openai/clip-vit-base-patch32`) |
| Agent Framework | LangChain |
| Image Processing | PIL (Pillow) |
| HTTP Client | Requests |
| Environment Management | python-dotenv |

---

## 🧪 Example Usage

### 🖊️ Text Query Example

> _"Sunset over mountains with orange sky"_ → Finds visually matching images.

### 🖼️ Image Query Example

Upload an image → Finds visually similar results from indexed store.

---

## 🧾 Notes

- The data is getting fetched and injected using **temporary API**.  
- Ensure Qdrant runs locally on port `6333`.  

---
