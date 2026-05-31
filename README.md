# Big Data Analyst for facebook


## Project Overview

This project analyzes a social media post dataset using Python. The main goal is to understand post performance, clean and prepare the data, create visualizations, apply clustering methods and build machine learning models for regression and classification.

The analysis focuses on engagement and reach related variables such as likes, shares, comments, impressions, reach and total interactions.

## Dataset

The dataset used in this project is:

```bash
data_15.csv
```

The dataset contains information about social media posts, including post type, post category, posting time, reach, impressions, engagement, likes, comments, shares and total interactions.

## Technologies Used

The project was implemented in Python using the following libraries:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Project Workflow

### 1. Data Loading

The dataset is loaded using `pandas`:

```python
data = pd.read_csv("data_15.csv")
```

The first rows of the dataset are displayed to understand the structure of the data.

### 2. Data Cleaning

The notebook checks for missing values in the dataset.

Missing numeric values are replaced using the median value of each numeric column. Missing non-numeric values are replaced using the mode.

Duplicate rows are also removed to keep the dataset clean.

### 3. Categorical Encoding

Categorical variables are prepared for machine learning.

The variables `Type` and `Category` are encoded using `LabelEncoder`.

The variable `Post Weekday` is converted using one-hot encoding.

This step allows machine learning models to work with categorical data.

### 4. Feature Scaling

Numerical columns are normalized using `MinMaxScaler`.

This scales the values between 0 and 1, which helps make the variables comparable and improves the performance of some machine learning algorithms.

### 5. Exploratory Data Analysis

Several visualizations are created to understand the dataset.

The project includes:

* Histograms of numerical features
* Boxplots of numerical features
* Scatter plot of likes vs shares
* Correlation heatmap
* Pair plot of selected features

These visualizations help show the distribution of the data, possible outliers and relationships between variables.

### 6. Correlation Analysis

A correlation heatmap is created to show relationships between numerical variables.

This helps identify which features are strongly related to each other. For example, reach, impressions, likes, shares and total interactions can show important relationships in post performance.

### 7. Clustering Analysis

Two clustering methods are applied:

* K-Means Clustering
* Agglomerative Clustering

Before clustering, selected features are standardized using `StandardScaler`.

The selected features include:

```python
Page total likes
Lifetime Post Total Reach
Lifetime Engaged Users
like
share
```

The Elbow Method is used to help choose the number of clusters. In this project, 3 clusters are used.

The clusters are visualized using scatter plots based on `Lifetime Post Total Reach` and `Lifetime Engaged Users`.

### 8. Regression Model

A Linear Regression model is used to predict:

```python
Lifetime Post Total Reach
```

The features used for regression include post timing, impressions, engagement metrics, comments, likes, shares and total interactions.

The dataset is split into training and testing sets using an 80/20 split.

The model is evaluated using:

```python
Mean Squared Error, MSE
```

This shows how far the predicted reach values are from the real values.

### 9. Classification Models

The classification task predicts the post:

```python
Type
```

Two classification models are used:

* Random Forest Classifier
* Support Vector Classifier, SVC

The dataset is split into training and testing sets using an 80/20 split.

The models are evaluated using:

* Accuracy score
* Classification report

The classification report includes precision, recall and F1-score.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Open the Project Folder

```bash
cd your-repository-name
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Add the Dataset

Make sure the dataset file is in the same folder as the notebook:

```bash
data_15.csv
```

### 5. Run the Notebook

Open the notebook:

```bash
jupyter notebook "ali abdelrahman(GZULOX).ipynb"
```

Then run the cells from top to bottom.

## Repository Structure

```bash
.
├── README.md
├── ali abdelrahman(GZULOX).ipynb
└── data_15.csv
```

## Main Results

The analysis shows that social media post performance can be studied using engagement and reach metrics.

The visualizations help explain the distribution of the data and show possible relationships between variables such as likes and shares.

The clustering models divide posts into groups based on engagement and reach behavior.

The regression model predicts lifetime post reach using engagement and post-related features.

The classification models predict the post type using performance and engagement variables.

## Conclusion

This project demonstrates a complete data analysis and machine learning workflow for social media post data.

The workflow includes data cleaning, preprocessing, visualization, clustering, regression, and classification. The results show how social media engagement data can be used to understand post performance and support better data-driven decisions.
