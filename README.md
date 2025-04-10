# SHL Assessment Recommendation System

This project is an intelligent recommendation system designed to suggest relevant SHL assessments based on natural language queries or job descriptions. Built for the Gen AI Task, it leverages semantic search, LLMs, and a Streamlit interface.

## Live Demo
Try it out on Hugging Face Spaces:  
[https://huggingface.co/spaces/absara1am/shl-assessment-recommendation-system](https://huggingface.co/spaces/absara1am/shl-assessment-recommendation-system)  
*(Replace "your-username" with your actual Hugging Face username.)*

## Features
- Takes natural language input (e.g., "Java developers, max 40 minutes").
- Recommends up to 10 SHL assessments with:
  - Assessment Name & URL
  - Remote Testing Support (Yes/No)
  - Adaptive/IRT Support (Yes/No)
  - Duration & Test Type
- Hosted as a Streamlit app for easy interaction.

## Project Structure
- `web_scraping.py`: Scrapes SHL’s product catalog to create `shl_assessments.csv`.
- `data_preprocessing.py`: Cleans data and generates `processed_assessments.csv`.
- `recommendation_engine.py`: Core logic with embedding, reranking, and LLM-based constraint parsing.
- `app.py`: Streamlit frontend with integrated recommender (final deployment version).
- `api.py`: FastAPI endpoint (unused in final deployment due to resource limits).
- `requirements.txt`: Dependencies list.
- `data/`: Directory for CSV files (not tracked; generate locally).

## Approach
1. **Data**: Scraped SHL catalog using `requests` and `BeautifulSoup`.
2. **Preprocessing**: Cleaned data with `pandas`, mapped test types, and created a searchable text field.
3. **Recommendation**:
   - Embedding: `sentence-transformers/all-mpnet-base-v2`.
   - Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Constraints: `mistralai/Mixtral-8x7B-Instruct-v0.1` via LangChain LCEL, with regex fallback.
   - Filters and boosts for skills/duration/test types.
4. **Deployment**: Integrated into Streamlit, hosted on Hugging Face Spaces (API skipped due to free-tier limits).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/shl-assessment-recommender.git
   cd shl-assessment-recommender

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Set Environment Variable**:
   ```bash
   export HF_API_KEY='your_hugging_face_api_key'
Get your API key from Hugging Face.

4. **Prepare Data**:
- Run web_scraping.py to generate shl_assessments.csv.
- Run data_preprocessing.py to create processed_assessments.csv in data/.

5. Run Locally:
   ```bash
   streamlit run app.py
Open http://localhost:8501 in your browser.

Note: The API version (api.py) requires uvicorn and can be run with uvicorn api:app --reload, but it’s not used in the final deployment.

## Requirements
- Python 3.9+
- Libraries: pandas, sentence-transformers, torch, streamlit, langchain-huggingface, beautifulsoup4, etc. (see requirements.txt).

## Limitations
- API not deployed due to free-tier resource constraints (e.g., Render couldn’t handle large models).
- Evaluation metrics (Mean Recall@3, MAP@3) not computed due to lack of a benchmark set.
