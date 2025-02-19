# RAG Sense

## Introduction

**RAG Sense** is an **agentic RAG (Retrieval-Augmented Generation) application** designed to empower enterprises with **context-aware AI-driven insights**. DataChRAG Senseime leverages **Pinecone** for vector storage, **LangChain** for advanced retrieval workflows, and multiple data sources including **Confluence pages, local file monitoring, and web-based knowledge** to enhance response accuracy. It provides a **seamless API interface** and a **React-based frontend**, making it easy to integrate into existing workflows.

---

## Installation Instructions

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Node.js 18+** (for frontend)
- **pip** (Python package manager)
- **virtualenv** (Python virtual environment)

### Step 1: Clone the Repository
```sh
 git clone https://github.com/gandhiraketla/datachime.git
```

### Step 2: Navigate to the Backend Directory
```sh
 cd ragsense/backend
```

### Step 3: Create and Activate a Virtual Environment
```sh
 python -m venv venv
```
```sh
 # Windows
 venv\Scripts\activate
```
```sh
 # macOS/Linux
 source venv/bin/activate
```

### Step 4: Install Dependencies
```sh
 pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Create a `.env` file in the `backend` directory and add the following:
```ini
CONFLUENCE_URL=
CONFLUENCE_USERNAME=
CONFLUENCE_API_TOKEN=
PERPLEXITY_API_KEY=
PERPLEXITY_MODEL_NAME=
PINECONE_HOST=
OPENAI_API_KEY=
PINECONE_INDEX=ragindex
PINECONE_API_KEY=
LOCAL_FOLDER_MONITOR_PATH=
```

### Step 6: Create Pinecone Index (via UI)
Follow these steps to create a Pinecone index using the Pinecone web UI:

1. Log in to your Pinecone account at [https://app.pinecone.io](https://app.pinecone.io).
2. Navigate to the **Indexes** tab from the dashboard.
3. Click on the **Create Index** button.
4. Fill in the following details:
   - **Index Name**: `ragindex`
   - **Dimension**: `1024` (suitable for multilingual-e5-large embeddings)
   - **Metric**: `cosine`
   - **Pods**: Choose an appropriate number of pods based on your use case (e.g., `1 pod` for development).
5. Click **Create Index**.
6. Once the index is created, copy the **Index Name**, **API Key**, and **Host URL**, and update the respective fields in your `.env` file.

---

### Step 7: Start the Backend Services
#### Start Folder Watcher
```sh
 python main.py
```
This process will start monitoring the folder defined in `LOCAL_FOLDER_MONITOR_PATH`.

#### Start the API Server
Open another terminal, navigate to the `backend/api` directory, and run:
```sh
 uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Frontend Setup
The frontend is built with **React.js**. Follow these steps to set up and run it:

### Step 1: Navigate to the Frontend Directory
```sh
 cd ../frontend
```

### Step 2: Install Dependencies
```sh
 npm install
```

### Step 3: Start the Frontend Application
```sh
 npm start
```
This will start the frontend server on `http://localhost:3000/`.

---

## Usage
Once both the backend and frontend services are running, you can interact with the **DataChime** application through its UI or API.
- Upload documents to the monitored folder to **automatically index data**.
- Use the API to query stored knowledge with enhanced RAG capabilities.
- Explore insights through the **React-based UI**.

---

## Adding a New Data Connector
To add a new data connector, develop the connector as per `connectors.DataSourceConnector` and make an entry in `config.connector_mapping.json`.

---

## Contributing
We welcome contributions! Please **fork the repository**, create a new branch for your changes, and submit a pull request.

---

## License
This project is **open-source** and available under the **MIT License**.

For any issues or questions, feel free to raise an issue on [GitHub](https://github.com/gandhiraketla/datachime/issues).