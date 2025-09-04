# 🌍 Climate Tweet Sentiment Predictor

An AI-powered web application that analyzes climate-related tweets and predicts their sentiment using an LSTM deep learning model. The app also provides **Nigeria-specific stakeholder recommendations** and suggested actions based on the detected sentiment.

This project was developed by **Kolaru Gideon Mosimiloluwa**.

👉 Live App: [climatewebapp.streamlit.app](https://climatewebapp.streamlit.app)

---

## 🚀 Features

* **Real-Time Prediction**

  * Enter a single tweet and get sentiment prediction: *negative, neutral, or positive*.
  * Confidence score displayed with clear interpretation.
  * Context-aware recommendations for stakeholders and actions.

* **Batch Processing**

  * Upload a CSV file containing multiple tweets (with a column named `tweet`).
  * Get predictions for all tweets, with cleaned text, predicted labels, and confidence values.
  * Download results as a CSV file.
  * View aggregated sentiment counts in a chart.

* **Recommendations Engine**

  * Nigeria-focused recommendations mapped to each sentiment:

    * **Negative** → Urgent response actions for FMEnv, NEMA, NGOs, and local communities.
    * **Neutral** → Monitoring, logging, and resource sharing.
    * **Positive** → Amplifying best practices and community-led initiatives.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **Deep Learning**: TensorFlow / Keras (LSTM model)
* **Preprocessing**: Pandas, NumPy, Regex
* **Serialization**: Pickle, JSON

---

## 📂 Project Structure

```
climate-tweet-sentiment/
│── app1.py                 # Main Streamlit app
│── lstm_sentiment.h5       # Trained LSTM sentiment model
│── tokenizer.pickle        # Tokenizer for preprocessing tweets
│── label_map.json          # Label mapping (if available)
│── requirements.txt        # Python dependencies
│── 2149217819.jpg          # Background image for app
│── README.md               # Documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/climate-tweet-sentiment.git
   cd climate-tweet-sentiment
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model files exist**

   * `lstm_sentiment.h5`
   * `tokenizer.pickle`
   * `label_map.json` (optional, defaults will be used if missing)

---

## ▶️ Running the App

Run the Streamlit app locally:

```bash
streamlit run app1.py
```

Then open in your browser at:
👉 `http://localhost:8501`

---

## 📊 Input Data Format

For **Batch Mode**, your CSV must include a column named:

```
tweet
```

✔ Example row:

```
The flood situation is getting worse every day in Lagos.
```

---

## 🧠 Sentiment Categories

| Sentiment    | Description                                                 | Example                                               |
| ------------ | ----------------------------------------------------------- | ----------------------------------------------------- |
| **Negative** | Reports of flooding, pollution, or disasters                | “The river has overflowed again, destroying farms.”   |
| **Neutral**  | General updates, news, or monitoring statements             | “Weather in Abuja seems stable this week.”            |
| **Positive** | Climate solutions, successful interventions, awareness wins | “Our solar project just provided light to 200 homes!” |

---

## 📝 Future Improvements

* Expand sentiment categories for finer granularity.
* Add real-time Twitter API integration.
* Extend stakeholder recommendations beyond Nigeria.
* Deploy advanced transformer models (e.g., BERT, RoBERTa).

---

## 👨‍💻 Author

* **Kolaru Gideon Mosimiloluwa**

---
