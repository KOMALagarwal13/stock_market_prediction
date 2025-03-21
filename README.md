# stock_market_prediction
 

## **Project Overview**  
The **Stock Price Prediction System** is a machine learning-based application that predicts future stock prices based on historical data. It uses an **LSTM (Long Short-Term Memory)** model trained on real-time stock data from Yahoo Finance. The system also features an interactive **Streamlit web app** for easy stock price forecasting and visualization.  

## **Features**  
✅ Fetches real-time stock data from Yahoo Finance  
✅ Uses deep learning (LSTM) for accurate time-series predictions  
✅ Provides a web-based interface for user-friendly interaction  
✅ Displays stock price trends and visualizations  

## **Technologies Used**  
- **Programming Languages**: Python  
- **Machine Learning**: TensorFlow, Keras, Scikit-learn  
- **Data Processing**: Pandas, NumPy, MinMaxScaler  
- **Visualization**: Matplotlib, Seaborn  
- **Web Development**: Streamlit  
- **Data Source**: Yahoo Finance API  

## **Installation & Setup**  
### **Step 1: Install Required Libraries**  
Ensure Python is installed, then install dependencies:  
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras streamlit yfinance
```  

### **Step 2: Train the Model**  
Run the training script to create the stock price prediction model:  
```bash
python train_model.py
```  

### **Step 3: Predict Stock Prices**  
Use the prediction script to forecast stock prices:  
```bash
python predict.py
```  

### **Step 4: Run the Web App**  
Launch the **Streamlit** web application to interact with the system:  
```bash
streamlit run app.py
```  

## **How It Works**  
1️⃣ **Data Collection** – Fetches historical stock prices from Yahoo Finance  
2️⃣ **Data Preprocessing** – Normalizes and structures data for training  
3️⃣ **Model Training** – Uses LSTM neural networks to learn stock patterns  
4️⃣ **Prediction** – Forecasts future stock prices based on past trends  
5️⃣ **Visualization** – Displays price trends and predicted values in a web UI   

## **Future Enhancements**  
🚀 Add sentiment analysis from news articles  
🚀 Improve accuracy using hybrid models (GRU + LSTM)  
🚀 Deploy the model as a cloud-based API  


