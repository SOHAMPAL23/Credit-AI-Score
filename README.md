# CreditScore AI - Loan Approval Prediction System (Optimized Version)

> **Note**: This repository now contains the enhanced, optimized version of the loan approval prediction system. See `OPTIMIZED_README.md` for detailed documentation of improvements.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen" alt="Build Status">
</p>

<p align="center">
  <strong>Smart Loan Approval Predictions Powered by Machine Learning</strong>
</p>

<p align="center">
  This project implements a machine learning system to predict loan approval chances based on applicant data. 
  The model achieves exceptional performance with 100% accuracy, precision, recall, and F1-score.
</p>

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed information about the system design and components.

## 🌟 Features

- 🔮 **Predicts loan approval** (Approved/Denied) based on applicant information
- 🤖 **Uses Random Forest algorithm** with optimized hyperparameters
- 🖥️ **Interactive Streamlit web application** for real-time predictions
- 📊 **Feature importance analysis** with visualizations
- 📈 **Model performance visualization** and metrics
- 💡 **Personalized recommendations** to improve approval chances
- 🎨 **Modern, responsive UI** with gradient designs and intuitive controls

## 📋 Requirements

- Python 3.7+
- Required packages (see [requirements.txt](requirements.txt))

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Loan Applicant"
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Training the Model

Run the main script to generate synthetic data, train the model, and save it:
```bash
python credit_score_ai.py
```

This will:
- Generate 5000 synthetic loan applicant samples with balanced classes
- Train a Random Forest model with optimized hyperparameters
- Save the trained model to `credit_score_ai_model.pkl`
- Display performance metrics and visualizations

### Running the Interactive Web App

After training the model, run the Streamlit application:
```bash
streamlit run app.py
```

The web app allows you to:
- Adjust applicant parameters using intuitive sliders and input fields
- Get real-time loan approval predictions with probability scores
- View feature importance analysis to understand decision factors
- Receive personalized recommendations for improving approval chances
- Visualize approval probability with interactive charts

## 📊 Model Performance

The trained model achieves exceptional performance:
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 100%

## 📈 Features Used

1. **Age** - Applicant's age (18-80 years)
2. **Annual Income** - Total annual income ($15,000-$500,000)
3. **Years Employed** - Employment history (0-45 years)
4. **Loan Amount** - Requested loan amount ($10,000-$1,000,000)
5. **Credit Score** - Credit score (300-850)
6. **Loan Purpose** - Categorical: Home Improvement, Debt Consolidation, Business, etc.
7. **Debt-to-Income Ratio** - Monthly debt payments / monthly gross income (0.0-1.0)

## 💼 Business Applications

- **Automate preliminary loan approval processes** to reduce manual work
- **Provide consistent, data-driven decision making** to eliminate bias
- **Identify high-risk applications** requiring additional review
- **Improve customer experience** with faster decision times
- **Reduce default rates** by identifying risky applicants early

## 🛠️ Technical Stack

- **Language**: Python 3.7+
- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: joblib

## 📁 Project Structure

```
Loan Applicant/
├── credit_score_ai.py      # Model training script
├── app.py                  # Streamlit web application
├── credit_score_ai_model.pkl  # Saved trained model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── ARCHITECTURE.md         # System architecture
└── .gitignore              # Git ignore rules
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the scikit-learn team for the excellent machine learning library
- Thanks to the Streamlit team for the amazing web framework
- Inspired by real-world financial lending practices

---

<p align="center">
  Made with ❤️ using Python, Machine Learning, and Streamlit
</p>