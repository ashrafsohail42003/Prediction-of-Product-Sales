# Sales Data Cleaning and Analysis

This project focuses on loading, cleaning, and performing basic analysis on a retail sales dataset using Python and the Pandas library. The objective is to prepare the dataset for further analysis and machine learning tasks such as sales prediction.

---

## Dataset Overview

The dataset contains **8,523 rows** and **12 columns** representing information about products, stores, and their corresponding sales.

### Dataset Features

- **Item_Identifier** – Unique product ID  
- **Item_Weight** – Weight of the product  
- **Item_Fat_Content** – Fat level of the product (Low Fat / Regular)  
- **Item_Visibility** – Percentage of product visibility in the store  
- **Item_Type** – Category of the product  
- **Item_MRP** – Maximum retail price of the product  
- **Outlet_Identifier** – Unique store ID  
- **Outlet_Establishment_Year** – Year the store was established  
- **Outlet_Size** – Size of the outlet  
- **Outlet_Location_Type** – Location tier of the outlet  
- **Outlet_Type** – Type of outlet (Supermarket / Grocery Store)  
- **Item_Outlet_Sales** – Sales of the product in the outlet  

---

## Technologies Used

- Python
- Pandas
- Jupyter Notebook / Google Colab

---

## Project Steps

### 1. Data Loading

The dataset was loaded using the Pandas library.

```python
import pandas as pd

fname = 'sales_predictions_2023.csv'
df = pd.read_csv(fname)
