# AI-Powered TV Series Analysis System

## Overview
This project focuses on building an AI-powered NLP system to analyze TV series scripts using state-of-the-art natural language processing (NLP) models. The system incorporates multiple AI models, data scraping techniques, and a user-friendly interface powered by Gradio. The goal is to extract insights from TV series, including character relationships, recurring themes, and chatbot-based interactions.

## Features
- **Text Classification**: Uses Hugging Face’s zero-shot classification model to identify key themes in a series.
- **Character Network Analysis**: Extracts character relationships using named entity recognition (NER) with Spacy.
- **Character Chatbot**: Implements a chatbot using Llama LLM to mimic TV series characters.
- **Data Scraping**: Uses Scrapy to collect subtitle and transcript datasets from online sources.
- **Transformer-Based Analysis**: Leverages modern transformer models for text processing and classification.
- **User Interface**: Provides an interactive web UI using Gradio.

## Dataset Details
- **Source**: Scraped subtitle and transcript datasets from publicly available online sources.
- **Structure**: Each dataset contains:
  - Dialogue text
  - Character metadata
  - Timestamped interactions
- **Preprocessing**: Data is cleaned, tokenized, and structured for machine learning models.

## Project Workflow
1. **Data Collection & Preprocessing**
   - Select a TV series (e.g., Naruto) for analysis.
   - Scrape data using Scrapy from online sources.
   - Prepare structured datasets (subtitles, transcripts, classification labels).
   
2. **Natural Language Processing (NLP) Implementation**
   - Train and fine-tune text classifiers for theme analysis.
   - Use Spacy for named entity recognition to map character interactions.
   - Develop a chatbot capable of engaging with users as TV series characters.

3. **Machine Learning Model Integration**
   - Implement zero-shot classification with Hugging Face transformers.
   - Process text data using Pandas and NumPy.
   - Apply deep learning models to extract insights.

4. **Web Interface Development**
   - Use Gradio to create an interactive web UI.
   - Deploy the model to allow users to analyze scripts dynamically.

5. **Deployment & Execution**
   - Run theme classification on TV series episodes.
   - Store and structure the extracted data.
   - Enable chatbot interactions using trained LLMs.

## Technologies Used
- **Programming Language**: Python
- **Frameworks/Libraries**:
  - Hugging Face Transformers
  - Spacy for NER
  - Scrapy for web scraping
  - Pandas & NumPy for data processing
  - Gradio for UI development
  - PyTorch for deep learning
  - Matplotlib & NetworkX for character network visualization

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-tv-series-analysis.git
   cd ai-tv-series-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run data scraping to collect subtitles and transcripts:
   ```bash
   python scraper.py
   ```
4. Execute the theme classification:
   ```bash
   python theme_classifier.py
   ```
5. Start the interactive chatbot interface:
   ```bash
   python chatbot.py
   ```
6. Launch the Gradio web UI:
   ```bash
   python gradio_app.py
   ```

## Evaluation Metrics
- **Accuracy**: Performance of text classification and theme detection models.
- **F1 Score**: Evaluation of named entity recognition (NER) performance.
- **BLEU Score**: Assessment of chatbot response quality compared to original dialogues.
- **Graph Metrics**: Measures such as centrality and clustering for character network analysis.

## Use Cases
- **Content Analysis**: Understanding recurring themes and sentiments in TV series.
- **Fan Engagement**: Interactive chatbot that mimics favorite characters.
- **Academic Research**: Analyzing character dynamics and storytelling structures.
- **Screenwriting Assistance**: Helping writers analyze and refine scripts using AI insights.

## Future Improvements
- Expand character network analysis with advanced relationship mapping.
- Integrate multi-language support for NLP models.
- Optimize chatbot responses using reinforcement learning.
- Enhance the UI with additional visualization features.
- Develop real-time subtitle analysis for live TV streaming.


## License
This project is licensed under the Apache License.