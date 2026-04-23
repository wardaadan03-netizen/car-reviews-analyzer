# 🚗 Car Reviews Analyzer with LLMs

A comprehensive NLP system for analyzing car reviews using **traditional machine learning** and **Large Language Models (OpenAI / Anthropic)**.

It converts raw customer reviews into structured insights such as sentiment, key features, summaries, and interactive visualizations.

---

# ✨ Features

* 📊 Sentiment analysis (traditional + LLM-based)
* 🔍 Aspect-based sentiment detection (performance, comfort, safety, etc.)
* 📈 Feature extraction and keyword analysis
* 📝 Automatic review summarization
* 📊 Interactive visual dashboards (Matplotlib + Plotly)
* 🤖 Support for OpenAI GPT and Anthropic Claude
* 🚗 Car-specific insights (ratings, models, manufacturers)
* 💾 Exportable reports and processed datasets

---

# ⚙️ Installation

## 1. Clone repository

```bash
git clone https://github.com/yourusername/car-reviews-analyzer.git
cd car-reviews-analyzer
```

---

## 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Setup

## Set API keys (if using LLMs)

### Windows (PowerShell)

```bash
setx OPENAI_API_KEY "your_openai_key"
setx ANTHROPIC_API_KEY "your_anthropic_key"
```

Restart terminal after setting variables.

---

# 🚀 Usage

## ▶️ Run with sample data

```bash
python main.py
```

---

## 📁 Run with custom dataset

```bash
python main.py --data-source path/to/your/reviews.csv
```

---

## ⚡ Run without LLM (faster mode)

```bash
python main.py --no-llm
```

---

## 🤖 Use Anthropic instead of OpenAI

```bash
python main.py --llm-model anthropic
```

---

# 📂 Input Data Format

Your CSV must contain:

| Column      | Description              |
| ----------- | ------------------------ |
| review_text | Customer review text     |
| car_model   | Car model name           |
| rating      | Rating (1–5)             |
| year        | Manufacturing year (opt) |

---

# 📊 Output

The system generates:

* Processed dataset (`data/processed/`)
* Sentiment analysis results
* Aspect score analysis
* JSON summary report
* Interactive visualizations

---

# 📈 Example Output

```text
Sentiment Distribution:
Positive: 45%
Neutral: 30%
Negative: 25%
```

**Top Keywords**

* Positive: great, smooth, comfortable, excellent
* Negative: cheap, slow, noise, cramped

**Best Aspects**

* performance
* safety
* technology

**Weak Aspects**

* fuel economy
* maintenance cost

---

# 🛠️ Tech Stack

* Python 3.12
* Pandas, NumPy
* NLTK / NLP tools
* Matplotlib, Seaborn, Plotly
* OpenAI API
* Anthropic Claude API

---

# 📁 Project Structure

```text
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
```

---

# 📜 License

MIT License
---

# 🚀 Project Highlights

This project demonstrates:

* End-to-end NLP pipeline design
* Real-world text analytics
* Integration of ML + LLMs
* Data visualization and reporting

