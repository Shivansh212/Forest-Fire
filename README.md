# 🔥 Forest Fire Risk Predictor

> **Predicting wildfire ignition before it happens — using 41 years of climate data and deep learning.**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-MVP%20%2F%20In%20Progress-yellow?style=for-the-badge)

---

## 📌 Problem Statement

Forest fires cause catastrophic damage to ecosystems, wildlife, and human lives every year. Traditional fire detection systems react **after** a fire has started — often too late to prevent large-scale destruction.

This project flips that model: instead of **detecting** fires, it **predicts** them.

By learning from over **41 years of historical weather and fire data** sourced from **Georgia Southern University**, this system identifies dangerous atmospheric patterns and outputs a risk score — giving fire management authorities the window they need to act **before** ignition.

---

## ✨ Features

- 🧠 **LSTM-based Deep Learning** — A Long Short-Term Memory (LSTM) Recurrent Neural Network captures complex temporal weather patterns to model fire risk over time
- 📂 **CSV / File-based Input** — Feed in weather observation data via structured CSV files for quick batch predictions
- 📊 **Risk Score Output** — Outputs a probability score indicating the likelihood of a fire-start event on a given day
- 📉 **Model Evaluation Metrics** — Includes Confusion Matrix, Recall, Precision, and other diagnostics to assess model performance
- 📅 **41 Years of Training Data** — Trained on a rich historical dataset spanning four decades, enabling the model to recognize seasonal and long-term fire patterns

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow, Keras (LSTM RNN) |
| Data Processing | Pandas, NumPy |
| Model Evaluation | Scikit-learn |
| Web Framework | Flask (`application.py` + `templates/`) |
| Packaging | `setup.py` (installable Python package) |
| Data Format | CSV |

---

## 📥 Input Features

The model is trained on the following weather and derived features:

| Feature | Description |
|---|---|
| `DATE` | Observation date |
| `PRECIPITATION` | Daily precipitation (mm) |
| `MAX_TEMP` | Maximum temperature (°C) |
| `MIN_TEMP` | Minimum temperature (°C) |
| `AVG_WIND_SPEED` | Average wind speed (km/h) |
| `TEMP_RANGE` | Daily temperature range (MAX - MIN) |
| `WIND_TEMP_RATIO` | Wind speed to temperature interaction ratio |
| `MONTH` | Month of observation |
| `SEASON` | Encoded season (Spring/Summer/Fall/Winter) |
| `DAY_OF_YEAR` | Ordinal day of year |
| `LAGGED_PRECIPITATION` | Prior-day precipitation (lag feature) |
| `LAGGED_AVG_WIND_SPEED` | Prior-day wind speed (lag feature) |
| `YEAR` | Year of observation |
| `FIRE_START_DAY` | Target label — 1 if a fire started, 0 otherwise |

---

## 📊 Model Performance

The model is evaluated with a focus on **Recall** — because in fire prediction, **missing a fire is far costlier than a false alarm**.

| Metric | Score |
|---|---|
| 🔥 Fire Recall | **88%** |
| 🎯 Precision | **54%** |

> **Interpretation:** The model correctly identifies **88% of actual fire-risk days** — making it a strong candidate for early warning use. The 54% precision reflects the expected class imbalance (fires are rare events), and is an active area of improvement.

---

## 📸 Screenshots & Visualizations

> *Screenshots of training loss curves, confusion matrix, and prediction outputs are available in the `/Image` folder.*

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Shivansh212/Forest-Fire.git
cd Forest-Fire

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the project as a package
pip install -e .

# 4. Run the training pipeline
python src/pipelines/Training_Pipeline.py

# 5. Launch the Flask web app
python application.py
```

> The app will be available at `http://localhost:5000`

---

## 📁 Project Structure

```
forest-fire-risk-predictor/
│
├── IMAGE/                          # Project screenshots & visuals
│
├── Notebook/                       # Exploratory Data Analysis
│   ├── data/                       # Raw data used in notebook
│   └── Forest_Fire_EDA.ipynb       # EDA & model prototyping notebook
│
├── artifacts/                      # Auto-generated model artifacts
│   ├── data.csv                    # Full processed dataset
│   ├── train.csv                   # Training split
│   ├── test.csv                    # Test split
│   ├── lstm_model.h5               # Trained LSTM model weights
│   └── preprocessor.pkl            # Fitted data preprocessor
│
├── src/                            # Core source package
│   ├── components/                 # Modular ML pipeline components
│   │   ├── Data_Ingestion.py       # Loads & splits raw data
│   │   ├── Data_Transformation.py  # Feature engineering & scaling
│   │   ├── Model_Trainer.py        # LSTM model training & evaluation
│   │   └── __init__.py
│   │
│   ├── pipelines/                  # End-to-end pipeline orchestration
│   │   ├── Training_Pipeline.py    # Runs full training workflow
│   │   ├── Prediction_Pipeline.py  # Loads model & runs inference
│   │   └── __init__.py
│   │
│   ├── Exception.py                # Custom exception handling
│   ├── Logger.py                   # Centralized logging
│   ├── Utils.py                    # Shared utility functions
│   └── __init__.py
│
├── templates/                      # Flask HTML templates (Web UI)
├── application.py                  # Flask app entry point
├── setup.py                        # Package installation config
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔮 Future Scope

This project is actively evolving. Planned improvements include:

- 🌐 **Web App Deployment** — Build an interactive dashboard using Streamlit or FastAPI for real-time predictions
- 🌦️ **Live Weather API Integration** — Connect to real-time weather APIs (OpenWeatherMap, NOAA) for on-demand risk scoring
- 🗺️ **Geographic Expansion** — Extend training data beyond Georgia to cover other high-risk fire regions globally
- 🎯 **Model Accuracy Improvements** — Experiment with attention mechanisms, ensemble models, and oversampling (SMOTE) to improve Precision
- 📱 **Mobile Alert System** — Push fire-risk alerts to emergency responders via a mobile notification pipeline

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- **Dataset:** Georgia Southern University — 41 years of historical weather and fire records
- **Inspiration:** The growing global urgency of wildfire prevention and climate resilience

---
