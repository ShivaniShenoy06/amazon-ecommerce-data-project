# amazon-ecommerce-data-project
Cleaned and standardized Amazon e-commerce product dataset. 
# Amazon Dataset Data Cleaning

This repository contains a comprehensive data cleaning workflow for an Amazon product dataset. The project demonstrates best practices in handling real-world, messy data and preparing it for analysis or modeling.

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

## Project Overview

This project demonstrates the process of cleaning and preparing an Amazon product dataset for analysis. The goal is to address common data quality issues and produce a dataset ready for further analytics or modeling.

## Dataset Description

- **Rows:** 1,000
- **Columns:** 55
- **Features:** Product details, prices, categories, reviews, and more.
- **Source:** [eCommerce Dataset Samples](https://github.com/luminati-io/eCommerce-dataset-samples/tree/main)

## Initial Data Exploration

The notebook begins with loading the dataset and examining its structure, data types, and missing values to identify key data quality issues.

## Data Cleaning Steps

### Handling Missing Values

- Identified and addressed missing values in key columns.
- Filled missing prices where possible and dropped columns with all null or redacted values.

### Removing Duplicates

- Checked for and removed duplicate rows to ensure data integrity.

### Standardizing Formats

- Cleaned and standardized string columns by removing unwanted characters and whitespace.
- Standardized price columns to float and parsed date columns.
- Standardized domain URLs.

### Feature Engineering

- Created new features such as `discount_percentage`, cleaned discount columns, and calculated discount amounts.
- Extracted year and month from timestamps.
- Split and cleaned category columns into hierarchical levels.
- Extracted and unified product weight data from multiple columns.

### Standardizing and Normalizing Categorical Values

- Normalized values in columns like `availability` (e.g., mapping all variants of "in stock" to "In Stock").
- Standardized currency codes (e.g., "USD", "INR").
- Standardized and cleaned category columns for consistency.

## Summary

- The Amazon product dataset has been thoroughly cleaned and standardized.
- Key steps included handling missing values, removing duplicates, standardizing categorical and currency columns, normalizing price and discount information, and extracting hierarchical category and weight data.
- The resulting dataset is now consistent, well-structured, and ready for further analysis.

---

**For full code and detailed explanations, see the [Amazon_dataset.ipynb](Amazon_dataset.ipynb) notebook.**
