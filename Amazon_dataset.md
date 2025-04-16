# Amazon Dataset Data Cleaning

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Initial Data Exploration](#initial-data-exploration)
4. [Data Cleaning Steps](#data-cleaning-steps)
    - [Handling Missing Values](#handling-missing-values)
    - [Removing Duplicates](#removing-duplicates)
    - [Standardizing Formats](#standardizing-formats)
    - [Feature Engineering](#feature-engineering)
    - [Standardizing and Normalizing Categorical Values](#standardizing-and-normalizing-categorical-values)
5. [Summary](#summary)


## 1. Project Overview
This notebook demonstrates the process of cleaning and preparing an Amazon product dataset for analysis. The goal is to address common data quality issues and produce a dataset ready for further analytics or modeling.


## 2. Dataset Description
- **Rows:** 1000
- **Columns:** 55
- **Features:** Product details, prices, categories, reviews, and more.
- **Source:** https://github.com/luminati-io/eCommerce-dataset-samples/tree/main


## 3. Initial Data Exploration
Let's load the data and examine its structure, types, and missing values.



```python
import numpy as np
import pandas as pd
import re

from matplotlib import pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")

# Load your data
df = pd.read_csv("amazon-products.csv")  # Update with your actual file path

# Preview the data
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>title</th>
      <th>seller_name</th>
      <th>brand</th>
      <th>description</th>
      <th>initial_price</th>
      <th>final_price</th>
      <th>currency</th>
      <th>availability</th>
      <th>reviews_count</th>
      <th>...</th>
      <th>root_bs_category</th>
      <th>bs_category</th>
      <th>bs_rank</th>
      <th>badge</th>
      <th>subcategory_rank</th>
      <th>amazon_choice</th>
      <th>images</th>
      <th>product_details</th>
      <th>prices_breakdown</th>
      <th>country_of_origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-08 00:00:00.000</td>
      <td>Saucony Men's Kinvara 13 Running Shoe</td>
      <td>Orv███tor███</td>
      <td>Saucony</td>
      <td>When it comes to lightweight speed, nothing cr...</td>
      <td>NaN</td>
      <td>"57.79"</td>
      <td>USD</td>
      <td>In Stock</td>
      <td>702</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-09 00:00:00.000</td>
      <td>Kishigo Premium Black Series Heavy Duty Unisex...</td>
      <td>Ama███.co███</td>
      <td>Kishigo</td>
      <td>The Kishigo Premium Black Series Heavy Duty Ve...</td>
      <td>NaN</td>
      <td>"28.5"</td>
      <td>USD</td>
      <td>In Stock</td>
      <td>916</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-04 00:00:00.000</td>
      <td>TWINSLUXES Solar Post Cap Lights Outdoor - Wat...</td>
      <td>Twi███uxe███</td>
      <td>TWINSLUXES</td>
      <td>Solar Post Cap Lights Waterproof LED Fence Pos...</td>
      <td>"49.99"</td>
      <td>"33.99"</td>
      <td>USD</td>
      <td>In Stock</td>
      <td>3178</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-06-09 00:00:00.000</td>
      <td>Accutire MS-4021B Digital Tire Pressure Gauge ...</td>
      <td>Cit███ran███Dir██████</td>
      <td>Accutire</td>
      <td>About this item Heavy duty construction and ru...</td>
      <td>1.795000000000000e+01</td>
      <td>1.795000000000000e+01</td>
      <td>USD</td>
      <td>In Stock</td>
      <td>8034</td>
      <td>...</td>
      <td>Automotive</td>
      <td>Tire Repair Tools</td>
      <td>50.00</td>
      <td>NaN</td>
      <td>[{"subcategory_name":"Automotive","subcategory...</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-01-16 00:00:00.000</td>
      <td>SAURA LIFE SCIENCE Adivasi Ayurvedic Neelgiri ...</td>
      <td>PRA███ EN███PRI███</td>
      <td>SAURA LIFE SCIENCE</td>
      <td>This extraordinary fusion is designed to nouri...</td>
      <td>"1299"</td>
      <td>"799"</td>
      <td>INR</td>
      <td>In stock</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>



### Data Types and Missing Values
Check the data types and count missing values in each column.



```python
df.info()
df.isnull().sum().sort_values(ascending=False).head(10)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 55 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   timestamp             1000 non-null   object 
     1   title                 1000 non-null   object 
     2   seller_name           826 non-null    object 
     3   brand                 999 non-null    object 
     4   description           998 non-null    object 
     5   initial_price         814 non-null    object 
     6   final_price           997 non-null    object 
     7   currency              1000 non-null   object 
     8   availability          994 non-null    object 
     9   reviews_count         1000 non-null   int64  
     10  categories            1000 non-null   object 
     11  asin                  1000 non-null   object 
     12  buybox_seller         926 non-null    object 
     13  number_of_sellers     984 non-null    float64
     14  root_bs_rank          999 non-null    float64
     15  answered_questions    969 non-null    float64
     16  domain                1000 non-null   object 
     17  images_count          1000 non-null   int64  
     18  url                   1000 non-null   object 
     19  video_count           1000 non-null   int64  
     20  image_url             995 non-null    object 
     21  item_weight           753 non-null    object 
     22  rating                1000 non-null   float64
     23  product_dimensions    933 non-null    object 
     24  seller_id             1000 non-null   object 
     25  date_first_available  921 non-null    object 
     26  discount              821 non-null    object 
     27  model_number          892 non-null    object 
     28  manufacturer          970 non-null    object 
     29  department            662 non-null    object 
     30  plus_content          1000 non-null   bool   
     31  upc                   99 non-null     object 
     32  video                 1000 non-null   bool   
     33  top_review            992 non-null    object 
     34  variations            742 non-null    object 
     35  delivery              948 non-null    object 
     36  features              818 non-null    object 
     37  format                445 non-null    object 
     38  buybox_prices         818 non-null    object 
     39  parent_asin           383 non-null    object 
     40  input_asin            373 non-null    object 
     41  ingredients           46 non-null     object 
     42  origin_url            241 non-null    object 
     43  bought_past_month     204 non-null    float64
     44  is_available          217 non-null    object 
     45  root_bs_category      106 non-null    object 
     46  bs_category           106 non-null    object 
     47  bs_rank               106 non-null    float64
     48  badge                 76 non-null     object 
     49  subcategory_rank      106 non-null    object 
     50  amazon_choice         106 non-null    object 
     51  images                52 non-null     object 
     52  product_details       0 non-null      float64
     53  prices_breakdown      0 non-null      float64
     54  country_of_origin     0 non-null      float64
    dtypes: bool(2), float64(9), int64(3), object(41)
    memory usage: 416.1+ KB
    




    country_of_origin    1000
    prices_breakdown     1000
    product_details      1000
    ingredients           954
    images                948
    badge                 924
    upc                   901
    root_bs_category      894
    amazon_choice         894
    subcategory_rank      894
    dtype: int64



## 4. Data Cleaning Steps
This section covers the main cleaning operations performed on the dataset.


#### a. Handling Missing Values

- Checked for missing values in all columns.
- Filled missing `initial_price` with `final_price` where possible.
- Dropped columns with all null values or redacted data (e.g., `product_details`, `prices_breakdown`, `country_of_origin`, `seller_nam`).
.
.



```python
# Check for missing values
print(df.isna().sum())

#missing_percentage = (df.isnull().sum() / len(df)) * 100
#print(missing_percentage)

#to fill the null values in initial price column with final price
df['initial_price'] = df['initial_price'].fillna(df['final_price'])
#df['initial_price'].isnull().sum()

df['final_price'].isnull().sum()

# Drop columns with all nulls or redacted and url's as the information is redundant in this project
df = df.drop(columns=['seller_name', 'product_details', 'prices_breakdown', 'country_of_origin', 'images', 'image_url', 'url', 'origin_url'])
```

    timestamp                  0
    title                      0
    seller_name              174
    brand                      1
    description                2
    initial_price            186
    final_price                3
    currency                   0
    availability               6
    reviews_count              0
    categories                 0
    asin                       0
    buybox_seller             74
    number_of_sellers         16
    root_bs_rank               1
    answered_questions        31
    domain                     0
    images_count               0
    url                        0
    video_count                0
    image_url                  5
    item_weight              247
    rating                     0
    product_dimensions        67
    seller_id                  0
    date_first_available      79
    discount                 179
    model_number             108
    manufacturer              30
    department               338
    plus_content               0
    upc                      901
    video                      0
    top_review                 8
    variations               258
    delivery                  52
    features                 182
    format                   555
    buybox_prices            182
    parent_asin              617
    input_asin               627
    ingredients              954
    origin_url               759
    bought_past_month        796
    is_available             783
    root_bs_category         894
    bs_category              894
    bs_rank                  894
    badge                    924
    subcategory_rank         894
    amazon_choice            894
    images                   948
    product_details         1000
    prices_breakdown        1000
    country_of_origin       1000
    dtype: int64
    

### b. Removing Duplicates

We check for and remove duplicate rows to ensure data integrity.



```python
# Check for duplicates
df.duplicated().sum()

# Remove duplicates if any
#df = df.drop_duplicates()

```




    np.int64(0)



#### c. Standardizing Formats

- Cleaned and standardized string columns by removing unwanted characters and whitespace.
- Standardized price columns to float.
- Standardized domain URLs and parsed date columns.
s.




```python
# Clean string columns
for col in ['variations', 'delivery', 'features', 'subcategory_rank']:
    df[col] = df[col].str.replace('[', '', regex=False)
    df[col] = df[col].str.replace(']', '', regex=False)
    df[col] = df[col].str.replace('{', '', regex=False)
    df[col] = df[col].str.replace('}', '', regex=False)
    df[col] = df[col].str.replace('"', '', regex=False)
    df[col] = df[col].str.strip()

# Standardize domain
df['domain'] = df['domain'].str.strip()
df['domain'] = df['domain'].apply(lambda x: f"https://{x}/" if x.startswith('www.') else x)

# Convert price columns from scientific/string notation to float
def clean_price(price):
    if pd.isna(price):
        return np.nan
    if isinstance(price, str):
        # Remove quotes and dollar signs if present
        return float(price.replace('"', '').replace('$', '').replace(',', '').replace('₹', '').replace('£', ''))
    return float(price)

 
# Apply the function to price columns
df['initial_price'] = df['initial_price'].apply(lambda x: clean_price(x) if pd.notna(x) else x)
df['final_price'] = df['final_price'].apply(lambda x: clean_price(x) if pd.notna(x) else x)

# Standardize string columns
df['title'] = df['title'].str.strip()
df['brand'] = df['brand'].str.strip()
df['availability'] = df['availability'].str.strip()
df['availability'] = df['availability'].replace({'In Stock': 'In Stock', 'In Stock.':'In Stock', 'In stock':'In Stock'})
df['categories'] = df['categories'].str.strip()

# Parse timestamp, date_first_available column
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date_first_available'] = pd.to_datetime(df['date_first_available'], errors='coerce')

pd.options.display.float_format = '{:.2f}'.format

```


```python
df['availability'].value_counts()
```




    availability
    In Stock                                       709
    Only 1 left in stock - order soon               49
    Only 2 left in stock - order soon               29
    Only 5 left in stock - order soon               15
    In stock Usually ships within 2 to 3 days.      13
                                                  ... 
    Only 14 left in stock - order soon.              1
    Only 9 left in stock (more on the way).          1
    Only 13 left in stock (more on the way)          1
    In stock. Usually ships within 3 to 4 days.      1
    Only 4 left in stock (more on the way)           1
    Name: count, Length: 72, dtype: int64



#### d. Feature Engineering

- Created new features such as `discount_percentage`, `discount_cleaned`, and `discount_amount`.
- Extracted year and month from the timestamp.
- Split and cleaned category columns into hierarchical levels.
- Cleaned and merged item weight information from multiple col
- To ensure consistency between `initial_price`, `final_price`, and `discount_cleaned`, we:
- Impute missing `final_price` using `initial_price` and `discount_cleaned` where possible.
- Fix discrepancies between the calculated and existing `final_price`.
- Update `discount_amount` after corrections.umns.



```python
# Discount percentage
df['discount_percentage'] = ((df['initial_price'] - df['final_price']) / df['initial_price']) * 100
df['discount_percentage'] = df['discount_percentage'].round(2)


# Clean discount column
def clean_discount(discount):
    if pd.isna(discount):
        return None
    match = re.search(r'(\d+)%', str(discount)) or re.search(r'$$(\d+)%$$', str(discount)) or re.search(r'(\d+(\.\d+)?)', str(discount))  
    # Extract numeric value
    return float(match.group(1)) if match else None

# Calculate discount amount
def calculate_discount_amount(row):
    if pd.notna(row['initial_price']) and pd.notna(row['final_price']):
        return row['initial_price'] - row['final_price']
    return None

df['discount_amount'] = df.apply(calculate_discount_amount, axis=1)
df['discount_cleaned'] = df['discount'].apply(clean_discount)


# Extract year and month from timestamp
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month

# Split categories into subcategories
df['categories'] = df['categories'].str.replace('[', '', regex=False)
df['categories'] = df['categories'].str.replace(']', '', regex=False)
df['categories'] = df['categories'].str.replace('"', '', regex=False)
df['main_category'] = df['categories'].str.split(',').str[0]
df['sub_category_1'] = df['categories'].str.split(',').str[1]
df['sub_category_2'] = df['categories'].str.split(',').str[2].fillna('')
df['sub_category_3'] = df['categories'].str.split(',').str[3].fillna('')
df['sub_category_1'] = df['sub_category_1'].str.strip()
df['sub_category_2'] = df['sub_category_2'].str.strip()
df['sub_category_3'] = df['sub_category_3'].str.strip()

```


```python
# Clean and merge item weight
def extract_weight_from_dimensions(dim_str):
    if not isinstance(dim_str, str):
        return None
    weight_patterns = [
        r'(\d+(\.\d+)?)\s*Pounds?',
        r'(\d+(\.\d+)?)\s*Ounces?',
        r'(\d+(\.\d+)?)\s*Grams',
        r'(\d+(\.\d+)?)\s*Kilograms',
        r'(\d+(\.\d+)?)\s*g\b'
    ]
    for pattern in weight_patterns:
        match = re.search(pattern, dim_str, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(0).lower()
            if 'pound' in unit:
                return value
            elif 'ounce' in unit:
                return value / 16
            elif 'gram' in unit or 'g' in unit:
                return value / 453.592
            elif 'kilogram' in unit:
                return value * 2.20462
    return None

def clean_item_weight(weight_str):
    if pd.isna(weight_str):
        return None
    match = re.search(r'(\d+(\.\d+)?)\s*(\w+)', str(weight_str))
    if match:
        value = float(match.group(1))
        unit = match.group(3).lower()
        if 'pound' in unit or unit == 'lbs':
            return value
        elif 'ounce' in unit or unit == 'oz':
            return value / 16
        elif 'gram' in unit or unit == 'g':
            return value / 453.592
        elif 'kilogram' in unit or unit == 'kg':
            return value * 2.20462
    return None

df['weight_from_dimensions'] = df['product_dimensions'].apply(extract_weight_from_dimensions)
df['cleaned_item_weight'] = df['item_weight'].apply(clean_item_weight)

def merge_weights(row):
    if pd.notna(row['cleaned_item_weight']):
        return row['cleaned_item_weight']
    elif pd.notna(row['weight_from_dimensions']):
        return row['weight_from_dimensions']
    return None

df['merged_item_weight'] = df.apply(merge_weights, axis=1)
df['final_item_weight'] = df['merged_item_weight'].apply(lambda x: f"{x:.2f} pounds" if pd.notna(x) else None)

```


```python
def fix_nan_final_price(row):
    if pd.isna(row['final_price']) and pd.notna(row['discount_cleaned']):
        return row['initial_price'] * (1 - row['discount_cleaned'] / 100)
    return row['final_price']

# To ensure consistency between `initial_price`, `final_price`, and `discount_cleaned`
def fix_final_price_discrepancies(row):
    if pd.notna(row['discount_cleaned']):
        calculated_final_price = row['initial_price'] * (1 - row['discount_cleaned'] / 100)
        # Check if there's a significant difference between calculated and existing final price
        if pd.isna(row['final_price']) or abs(row['final_price'] - calculated_final_price) > 0.01:
            return calculated_final_price
    return row['final_price']

# Apply both fixes
df['final_price'] = df.apply(fix_nan_final_price, axis=1)
df['final_price'] = df.apply(fix_final_price_discrepancies, axis=1)

# Update discount_amount after fixing final_price
df['discount_amount'] = df['initial_price'] - df['final_price']

```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 60 columns):
     #   Column                  Non-Null Count  Dtype         
    ---  ------                  --------------  -----         
     0   timestamp               1000 non-null   datetime64[ns]
     1   title                   1000 non-null   object        
     2   brand                   999 non-null    object        
     3   description             998 non-null    object        
     4   initial_price           1000 non-null   float64       
     5   final_price             1000 non-null   float64       
     6   currency                1000 non-null   object        
     7   availability            994 non-null    object        
     8   reviews_count           1000 non-null   int64         
     9   categories              1000 non-null   object        
     10  asin                    1000 non-null   object        
     11  buybox_seller           926 non-null    object        
     12  number_of_sellers       984 non-null    float64       
     13  root_bs_rank            999 non-null    float64       
     14  answered_questions      969 non-null    float64       
     15  domain                  1000 non-null   object        
     16  images_count            1000 non-null   int64         
     17  video_count             1000 non-null   int64         
     18  item_weight             753 non-null    object        
     19  rating                  1000 non-null   float64       
     20  product_dimensions      933 non-null    object        
     21  seller_id               1000 non-null   object        
     22  date_first_available    819 non-null    datetime64[ns]
     23  discount                821 non-null    object        
     24  model_number            892 non-null    object        
     25  manufacturer            970 non-null    object        
     26  department              662 non-null    object        
     27  plus_content            1000 non-null   bool          
     28  upc                     99 non-null     object        
     29  video                   1000 non-null   bool          
     30  top_review              992 non-null    object        
     31  variations              742 non-null    object        
     32  delivery                948 non-null    object        
     33  features                818 non-null    object        
     34  format                  445 non-null    object        
     35  buybox_prices           818 non-null    object        
     36  parent_asin             383 non-null    object        
     37  input_asin              373 non-null    object        
     38  ingredients             46 non-null     object        
     39  bought_past_month       204 non-null    float64       
     40  is_available            217 non-null    object        
     41  root_bs_category        106 non-null    object        
     42  bs_category             106 non-null    object        
     43  bs_rank                 106 non-null    float64       
     44  badge                   76 non-null     object        
     45  subcategory_rank        106 non-null    object        
     46  amazon_choice           106 non-null    object        
     47  discount_percentage     997 non-null    float64       
     48  discount_amount         1000 non-null   float64       
     49  discount_cleaned        821 non-null    float64       
     50  year                    1000 non-null   int32         
     51  month                   1000 non-null   int32         
     52  main_category           1000 non-null   object        
     53  sub_category_1          1000 non-null   object        
     54  sub_category_2          1000 non-null   object        
     55  sub_category_3          1000 non-null   object        
     56  weight_from_dimensions  278 non-null    float64       
     57  cleaned_item_weight     753 non-null    float64       
     58  merged_item_weight      951 non-null    float64       
     59  final_item_weight       951 non-null    object        
    dtypes: bool(2), datetime64[ns](2), float64(14), int32(2), int64(3), object(37)
    memory usage: 447.4+ KB
    

### e. Standardizing and Normalizing Categorical Values

To ensure consistency and usability, we standardize categorical columns such as `availability`, `currency`, and `categories`.

#### Standardize Availability
We normalize all variants of "in stock" and similar phrases to "In Stock".



```python
def standardize_availability(val):
    if pd.isna(val):
        return None
    val = val.strip().lower()
    if "in stock" in val:
        return "In Stock"
    elif "out of stock" in val:
        return "Out of Stock"
    elif "unavailable" in val:
        return "Unavailable"
    else:
        return val.title()  # fallback: capitalize first letter

df['availability'] = df['availability'].apply(standardize_availability)

```


```python
df['availability'].value_counts()
```




    availability
    In Stock                              982
    Usually Ships Within 1 To 2 Months      2
    Usually Ships Within 6 To 10 Days.      2
    Available To Ship In 1-2 Days           2
    Usually Ships Within 7 To 8 Days        1
    Usually Ships Within 7 Days.            1
    Usually Ships Within 8 To 9 Days        1
    Usually Ships Within 7 Days             1
    Usually Ships Within 1 To 2 Weeks       1
    Usually Ships Within 2 To 3 Days        1
    Name: count, dtype: int64



## 5. Summary

- The Amazon product dataset has been thoroughly cleaned and standardized.
- Key data cleaning steps included:
  - Handling missing values and removing duplicates.
  - Standardizing categorical columns such as `availability`, `currency`, and hierarchical `category` columns.
  - Parsing and normalizing price and discount information.
  - Extracting and unifying product weight data.
  - Splitting and cleaning category columns into `main_category`, `sub_category_1`, `sub_category_2`, and `sub_category_3` for better analysis.
- The resulting dataset is now consistent, well-structured, and ready for further analysis.


```python

```
