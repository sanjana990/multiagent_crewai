# Multi-Agent CrewAI System

This project implements a multi-agent AI system using CrewAI and LangChain to process user queries. It breaks down queries into steps, researches using tools and memory, and analyzes to provide clear answers. The system leverages Cohere for language modeling, FAISS for vector storage/memory, and SerpAPI for web searches.

## Features
- **Task Decomposition**: A Planner agent breaks down user queries.
- **Research with Tools**: A Researcher agent uses web search and long-term memory (FAISS) to gather information.
- **Analysis and Response**: An Analyzer agent summarizes and provides a final answer.
- **Persistent Memory**: Stores and retrieves information using FAISS vector store.

## Prerequisites
- Python 3.8+
- API Keys:
  - Cohere API Key (for LLM and embeddings)
  - SerpAPI Key (for web searches)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/sanjana990/multiagent_crewai.git
   cd multiagent_crewai
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables: Create a `.env` file in the root directory with your keys:
   ```
   COHERE_API_KEY=your_cohere_api_key
   SERPAPI_API_KEY=your_serpapi_api_key
   ```

## Usage
Run the main script:

- You'll be prompted to enter a query (e.g., "What's the latest news on AI advancements?").
- The system will process it through the agents and output a final answer.

## Project Structure
- `main.py`: Core script defining agents, tasks, and crew execution.
- `requirements.txt`: List of Python dependencies.
- `faiss_index/`: Directory for FAISS vector store (generated/loaded during runtime).
- `faiss_memory_store/`: Additional memory storage (if applicable).

## Contributing
Feel free to fork the repo and submit pull requests. For issues, open a ticket on GitHub.
