🚗 Car Reviews Analyzer with LLMs

A comprehensive NLP-based system for analyzing car reviews using traditional machine learning techniques and Large Language Models (OpenAI / Anthropic).

It transforms raw customer reviews into structured insights, including sentiment analysis, feature extraction, summaries, and visual dashboards.

✨ Features
📊 Sentiment analysis (traditional + LLM-based)
🔍 Aspect-based sentiment detection (performance, comfort, safety, etc.)
📈 Feature extraction and keyword analysis
📝 Automated review summarization
📊 Interactive visualizations (Matplotlib + Plotly)
🤖 Support for OpenAI GPT and Anthropic Claude
🚗 Car-specific insights (ratings, models, manufacturers)
💾 Export processed results and reports
📦 Installation
1. Clone the repository
git clone https://github.com/yourusername/car-reviews-analyzer.git
cd car-reviews-analyzer
2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux
3. Install dependencies
pip install -r requirements.txt
🔐 Environment Setup

For LLM features, set your API keys:

Windows (PowerShell)
setx OPENAI_API_KEY "your_openai_key"
setx ANTHROPIC_API_KEY "your_anthropic_key"

Restart terminal after setting variables.

🚀 Usage
▶️ Run with sample data
python main.py
📁 Run with your dataset
python main.py --data-source path/to/your/reviews.csv
⚡ Disable LLM (faster mode)
python main.py --no-llm
🤖 Use Anthropic instead of OpenAI
python main.py --llm-model anthropic
📂 Input Data Format

Your CSV file must contain:

Column	Description
review_text	Customer review text
car_model	Car model name
rating	Rating (1–5)
year	Manufacturing year (optional)
📊 Output

The system generates:

Processed dataset (data/processed/)
Sentiment analysis results
Aspect scores
JSON summary report
Visual charts and dashboards
📈 Example Output
Sentiment Distribution:
- Positive: 45%
- Neutral: 30%
- Negative: 25%

Top Keywords:
Positive: great, smooth, comfortable, excellent
Negative: cheap, slow, noise, cramped

Best Aspects:
- performance
- safety
- technology

Weak Aspects:
- fuel economy
- maintenance cost
🛠️ Tech Stack
Python 3.12
Pandas, NumPy
NLTK / NLP tools
Matplotlib, Seaborn, Plotly
OpenAI API
Anthropic Claude API
📌 Project Structure
car-review-analyzer/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── sentiment_analyzer.py
│   ├── feature_extractor.py
│   ├── llm_analyzer.py
│   ├── summarizer.py
│   └── visualizer.py
│
├── main.py
├── config.py
├── requirements.txt
└── README.md
⚙️ Running the Project
pip install -r requirements.txt
python main.py
📜 License

MIT License

🚀 Final Note

This project demonstrates:

End-to-end NLP pipeline design
Real-world text analytics
Integration of traditional ML + modern LLMs
Data visualization and reporting