# Stock Market Price Prediction System Using Machine Learning

This project leverages **Machine Learning**, specifically **LSTM (Long Short-Term Memory)** networks, to predict short-term stock price movements. By integrating real-time data, it provides both **actual stock prices** and **predicted stock trends** for the next 30 days, enabling investors to make more informed decisions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Overview

The **Stock Market Prediction System** uses **LSTM neural networks** to predict stock prices for the next 30 days based on **historical price data**. The system fetches **real-time data** from **Yahoo Finance API** and dynamically updates predictions. A user-friendly interface is provided to visualize actual vs. predicted stock prices using **Chart.js**.

---

## Features

- **Real-Time Stock Data**: Fetches historical and real-time stock prices.
- **Prediction for the Next 30 Days**: Predicts short-term trends using LSTM models.
- **Data Visualization**: Displays actual and predicted prices using interactive **Chart.js** visualizations.
- **Model Retraining**: Dynamic retraining to improve prediction accuracy with new data.
- **User-Friendly Interface**: An intuitive dashboard for both novice and experienced investors.

---

## Technology Stack

- **Frontend**: 
  - HTML, CSS, **JavaScript** (for user interface and real-time visualizations)
  - **Chart.js** (for data visualizations)
  
- **Backend**: 
  - **Flask** (for the backend server and API handling)
  - **Python** (for data processing and machine learning)
  - **yFinance** (for fetching stock data)
  - **TensorFlow / Keras** (for training LSTM models)
  
- **Database**:
  - No database required for this project. All data is fetched dynamically via the **Yahoo Finance API**.

---

## Installation

Follow these steps to run the project locally.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/iamDeepakHR/Stock-Price-Prediction-Using-Machine-Learning.git
   cd Stock-Price-Prediction-Using-Machine-Learning
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the necessary model and scaler files**:
   - Place the trained LSTM model (`stock_lstm_model.keras`) and scaler (`scaler.pkl`) into the `model/` directory.

---

## Usage

1. **Run the Flask app**:
   ```bash
   python app.py
   ```

2. **Access the app**:
   Open your browser and navigate to `http://127.0.0.1:5000/`.

3. **Predict Stock Prices**:
   - Enter a stock symbol (e.g., `AAPL`, `GOOG`, etc.) in the input field and click "Predict."
   - View the **actual stock data** and **predicted stock prices** for the next 30 days on the interactive graph.

4. **Train the Model**:
   - If you want to retrain the model, click the "Train Model" button to trigger training on new data.
   - The model will be retrained with the latest stock data and saved for future use.

---

## Contributing

We welcome contributions to enhance the project further. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -m 'Add feature'`).
4. Push to your forked repository (`git push origin feature-name`).
5. Submit a pull request for review.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Enjoy using the **Stock Market Prediction System** and happy investing!
