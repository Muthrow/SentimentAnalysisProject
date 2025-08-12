# SentimentAnalysisProject
 
python 3.11 for stability and compatibility with TOML files (streamlit secrets)

# Social Media Sentiment Analyzer (Proof of Concept)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zp-sentimentanalysis.streamlit.app/)

A proof-of-concept Streamlit application designed to fetch and analyze sentiment from various social media platforms using Hugging Face Transformers.

---

## About This Project

This project was created to explore and demonstrate the feasibility of building a real-time sentiment analysis tool. It connects to platform APIs (like YouTube, Twitter, and Reddit), pulls recent data based on user input, and performs sentiment analysis on the text content.

**Please Note:** The current version is a functional **proof of concept**. The primary goal was to validate the data fetching and analysis pipeline. The code is not yet optimized or refactored for stability or scalability.

---

## Features

* **Analyze YouTube Comments:** Enter a YouTube video URL to fetch and analyze the sentiment of its top comments.
* **Analyze Reddit Threads:** Enter a Reddit post URL to analyze the comments.
* **Analyze Twitter Posts:** Fetch data from Twitter to perform sentiment analysis.
* **Analyze Plain Text:** Fetch data from Twitter to perform sentiment analysis.
* **Powered by Transformers:** Uses a `distilbert-base-uncased` model from Hugging Face for sentiment analysis.
* **Basic Visualization:** Displays results in data tables and simple charts.

---

## Setup and Usage

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Muthrow/SentimentAnalysisProject.git](https://github.com/Muthrow/SentimentAnalysisProject.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    Create a file at `.streamlit/secrets.toml` and add your API keys. Copy the following template and fill in your keys:
```toml

    # .streamlit/secrets.toml

    # Twitter/X API
T_BEARER_TOKEN = 
T_ACCESS_TOKEN = 
T_ACCESS_TOKEN_SECRET = 
T_CONSUMER_KEY = 
T_CONSUMER_SECRET = 

    # YouTube/Google API
G_API_KEY = 

    # Reddit API
R_CLIENT_ID = 
R_CLIENT_SECRET = 
R_USER_AGENT = 
```

    Make sure to insert your actual API keys.


5.  **Run the app:**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## Project Status & Roadmap

**Current Status:** Proof of Concept

The current implementation uses separate, duplicated scripts for each social media platform. While functional, this approach is not maintainable.

**Future Work:**
The next phase of this project will involve a complete refactor to evolve it into a stable and scalable application. Planned improvements include:

* **Modular Codebase:** Refactor the separate scripts into a unified, modular structure.
* **Performance Optimization:** Implement robust caching for models and data to improve speed and reduce API calls.
* **Improved UI/UX:** Enhance the user interface with a cleaner layout and more advanced Streamlit components.
* **Robust Dependency Management:** Clean up `requirements.txt` and manage dependencies professionally.
* **Developer Tooling:** Integrate modern development tools (`ruff`, `pytest`, `pre-commit`) to ensure code quality and reliability.