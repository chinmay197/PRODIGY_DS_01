#  PRODIGY_DS_01 – House Price Prediction Using Linear Regression

This project is part of my internship at **Prodigy InfoTech** under the Data Science track.  
The goal is to implement a **Linear Regression model** to predict house prices based on key features like square footage, number of bedrooms, and bathrooms.


#  Problem Statement

To predict the **sale price of a house** based on:
- `GrLivArea`: Above-ground living area (in square feet)
- `BedroomAbvGr`: Number of bedrooms above ground
- `FullBath`: Number of full bathrooms


# Dataset

The dataset used in this project is from the Kaggle competition:  
**House Prices – Advanced Regression Techniques**  
🔗 [Kaggle Dataset Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

File used:
- `train.csv` (renamed to `house_data.csv`)


# Key Learnings

- Data loading and exploration using **pandas**
- Feature selection from a real-world dataset
- Implementing and training **Linear Regression** with **scikit-learn**
- Evaluating model performance using **R² Score** and **RMSE**
- Visualizing results with **matplotlib** and **seaborn**


# Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


# Model Evaluation

After training and testing the model:

- **R² Score**: ~0.66 (may vary slightly)
- **RMSE**: Varies depending on feature scaling and dataset split
- A scatter plot was generated to compare **Actual vs Predicted Sale Prices**



