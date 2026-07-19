# Car Price Prediction using Machine Learning

A machine learning project that predicts the selling price of used cars using **Linear Regression** and **Lasso Regression**. The project performs data preprocessing, feature encoding, model training, and evaluation using the CarDekho dataset.

---

## Project Overview

The objective of this project is to build a regression model that can estimate the selling price of a used car based on features such as:

- Manufacturing Year
- Present Price
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission
- Number of Previous Owners

---

## Dataset

**Dataset:** `car data.csv`

### Features

| Column | Description |
|---------|-------------|
| Car_Name | Name of the car |
| Year | Manufacturing year |
| Selling_Price | Selling price (Target Variable) |
| Present_Price | Current ex-showroom price |
| Kms_Driven | Kilometers driven |
| Fuel_Type | Petrol/Diesel/CNG |
| Seller_Type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

Dataset Size:

- Rows: **301**
- Columns: **9**

---

## Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Project Workflow

### 1. Import Libraries

- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### 2. Load Dataset

```python
car_dataset = pd.read_csv("car data.csv")
```

### 3. Data Exploration

- Display first 5 rows
- Check shape
- Check data types
- Check missing values
- View categorical distributions

### 4. Data Preprocessing

Categorical columns are encoded as numerical values.

Fuel Type

| Category | Encoding |
|----------|----------|
| Petrol | 0 |
| Diesel | 1 |
| CNG | 2 |

Seller Type

| Category | Encoding |
|----------|----------|
| Dealer | 0 |
| Individual | 1 |

Transmission

| Category | Encoding |
|----------|----------|
| Manual | 0 |
| Automatic | 1 |

### 5. Feature Selection

Input Features:

- Year
- Present_Price
- Kms_Driven
- Fuel_Type
- Seller_Type
- Transmission
- Owner

Target:

- Selling_Price

### 6. Train-Test Split

```python
train_test_split(
    X,
    Y,
    test_size=0.1,
    random_state=2
)
```

- Training Data: 90%
- Testing Data: 10%

### 7. Model Training

Models used:

- Linear Regression
- Lasso Regression

Example:

```python
model = LinearRegression()
model.fit(X_train, Y_train)
```

### 8. Model Evaluation

Metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

---

## Project Structure

```
Car-Price-Prediction/
│
├── car data.csv
├── Car_Price_Prediction.ipynb
├── README.md
└── requirements.txt
```

---

## Results

The trained model predicts the selling price of a used car based on the selected features.

Performance is evaluated using:

- R² Score
- Mean Absolute Error
- Mean Squared Error

---

## Possible Improvements

- One-Hot Encoding instead of Label Encoding
- Feature Engineering
- Hyperparameter Tuning
- Cross Validation
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Model Deployment using Streamlit or Flask

---

## Common Error

### Error

```
ValueError: could not convert string to float: 'Petrol'
```

### Cause

One or more categorical columns still contain string values.

### Solution

Check the data types before training:

```python
print(X.dtypes)
```

If `Fuel_Type`, `Seller_Type`, or `Transmission` are still of type `object`, encode them before calling `fit()`.

Example:

```python
car_dataset.replace({
    'Fuel_Type': {'Petrol':0, 'Diesel':1, 'CNG':2},
    'Seller_Type': {'Dealer':0, 'Individual':1},
    'Transmission': {'Manual':0, 'Automatic':1}
}, inplace=True)

car_dataset = car_dataset.infer_objects(copy=False)
```

Alternatively, use `map()`:

```python
car_dataset["Fuel_Type"] = car_dataset["Fuel_Type"].map({
    "Petrol":0,
    "Diesel":1,
    "CNG":2
})
```

---

## Future Scope

- Build a web application using Streamlit
- Deploy on Render or Hugging Face Spaces
- Add real-time user input for prediction
- Save and load trained models using Pickle

---

## Author

**Vedant Kalambe**

B.Tech Electronics and Telecommunication Engineering

Machine Learning Enthusiast
