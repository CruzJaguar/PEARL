# PEARL
Well Goodluck To Us I guess
# House Price Prediction
# House Price Prediction


Predict house prices using regression models (Linear, Ridge, Lasso, RandomForest, XGBoost) and deploy a Streamlit app.


## Setup


1. Create virtual environment:
python -m venv venv
# Activate
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate


2. Install dependencies:
pip install -r requirements.txt


3. Add dataset to data/housing.csv (optional). If none, uses sklearn California housing.


4. Train models:
python src/train.py


5. Run Streamlit app:
streamlit run app/app.py
