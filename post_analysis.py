"""
Social Media Post Performance Analysis
Converted from Jupyter Notebook: ali abdelrahman(GZULOX).ipynb
Author: Ali Abdelrahman (GZULOX)

This script loads data_15.csv, performs data cleaning, visualization,
clustering, regression, and classification analysis.

Required dataset file: data_15.csv
"""


# %% Cell 1
import pandas as pd
data = pd.read_csv('data_15.csv')
data.head()
# Ali abdelrahman(GZULOX) 

# %% Cell 2
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns
data[non_numeric_cols] = data[non_numeric_cols].fillna(data[non_numeric_cols].mode().iloc[0])

data.drop_duplicates(inplace=True)

data.head()

# %% Cell 3
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'])
data['Category'] = label_encoder.fit_transform(data['Category'])

data = pd.get_dummies(data, columns=['Post Weekday'], drop_first=True)

data.head()

# %% Cell 4
from sklearn.preprocessing import MinMaxScaler

numerical_cols = ['Page total likes', 'Post Hour', 'Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
                  'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                  'Lifetime Post Impressions by people who have liked your Page',
                  'Lifetime Post reach by people who like your Page',
                  'Lifetime People who have liked your Page and engaged with your post',
                  'comment', 'like', 'share', 'Total Interactions']

scaler = MinMaxScaler()

data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

data.head()

# %% Cell 5
import matplotlib.pyplot as plt

numerical_cols = ['Page total likes', 'Post Hour', 'Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
                  'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                  'Lifetime Post Impressions by people who have liked your Page',
                  'Lifetime Post reach by people who like your Page',
                  'Lifetime People who have liked your Page and engaged with your post',
                  'comment', 'like', 'share', 'Total Interactions']

data[numerical_cols].hist(bins=20, figsize=(15, 15))
plt.suptitle('Histograms of Numerical Features', y=1.02)
plt.show()
#Explanation: Histograms display the frequency distribution of each numerical feature. 
#This helps to understand the range, skewness, and presence of outliers in the data. 
#For example, we can see if the distribution of Lifetime Post Total Reach is right-skewed or if Page total likes follows a normal distribution.

# %% Cell 6
plt.figure(figsize=(15, 10))
data[numerical_cols].boxplot()
plt.xticks(rotation=90)
plt.title('Box Plots of Numerical Features')
plt.show()

#Explanation: Box plots summarize the distribution of numerical data through their quartiles. 
#They highlight the median, upper, and lower quartiles, as well as potential outliers. 
#This can be useful for identifying which features have a large spread or significant outliers, such as the number of Total Interactions.

# %% Cell 7
plt.figure(figsize=(10, 6))
plt.scatter(data['like'], data['share'], alpha=0.5)
plt.title('Scatter Plot of Likes vs Shares')
plt.xlabel('Likes')
plt.ylabel('Shares')
plt.show()
#Explanation: Scatter plots allow us to visualize the relationship between two numerical features. 
#For instance, the scatter plot of like vs share helps to identify if there's a correlation between the number of likes and shares a post receives. 
#A clear pattern would suggest a relationship between these two features.

# %% Cell 8
import seaborn as sns
correlation_matrix = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Feature Correlations')
plt.show()

#Explanation: A heatmap displays the correlation matrix, which shows the Pearson correlation coefficients between pairs of features. 
# This is useful for identifying highly correlated features that might be redundant. 
#For example, if Lifetime Post Total Reach and Lifetime Post Total Impressions are highly correlated, one of them might be removed from the model to reduce multicollinearity.

# %% Cell 9
subset = ['Page total likes', 'Lifetime Post Total Reach', 'Lifetime Post Total Impressions', 'like', 'share']

sns.pairplot(data[subset])
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

#Explanation: Pair plots provide a grid of scatter plots for each pair of features, along with the distribution of individual features on the diagonal. 
#This comprehensive visualization helps to understand the relationships and interactions between multiple features simultaneously. 
#For example, we can observe if posts with higher Lifetime Post Total Reach also receive more likes and shares.

# %% Cell 10
from sklearn.preprocessing import StandardScaler
import pandas as pd
features = ['Page total likes', 'Lifetime Post Total Reach', 'Lifetime Engaged Users', 'like', 'share']
data_selected = data[features]

data_selected.fillna(data_selected.median(), inplace=True)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# %% Cell 11
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

data['KMeans_Cluster'] = kmeans_labels

#Explanation: The Elbow method helps determine the optimal number of clusters by plotting the inertia (within-cluster sum of squares) against the number of clusters.
# The point where the curve bends (elbow) indicates the optimal number of clusters. In this case, we choose 3 clusters.


# %% Cell 12
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(data_scaled)


data['Agglomerative_Cluster'] = agg_labels


fig, axes = plt.subplots(1, 2, figsize=(16, 6))


sns.scatterplot(ax=axes[0], x='Lifetime Post Total Reach', y='Lifetime Engaged Users', hue='KMeans_Cluster', data=data, palette='Set1')
axes[0].set_title('K-Means Clustering')
axes[0].set_xlabel('Lifetime Post Total Reach')
axes[0].set_ylabel('Lifetime Engaged Users')


sns.scatterplot(ax=axes[1], x='Lifetime Post Total Reach', y='Lifetime Engaged Users', hue='Agglomerative_Cluster', data=data, palette='Set1')
axes[1].set_title('Agglomerative Clustering')
axes[1].set_xlabel('Lifetime Post Total Reach')
axes[1].set_ylabel('Lifetime Engaged Users')

plt.tight_layout()
plt.show()

#Explanation: The scatter plots show the clusters formed by K-Means and Agglomerative Clustering.
# Both plots use Lifetime Post Total Reach and Lifetime Engaged Users as the x and y axes, respectively 
## to visualize how the data points are grouped into clusters by each algorithm.


# %% Cell 13
data.head()

# %% Cell 14
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


regression_target = 'Lifetime Post Total Reach'
regression_features = ['Page total likes', 'Post Hour', 'Lifetime Post Total Impressions',
                      'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                      'Lifetime Post Impressions by people who have liked your Page',
                      'Lifetime Post reach by people who like your Page',
                      'Lifetime People who have liked your Page and engaged with your post',
                      'comment', 'like', 'share', 'Total Interactions']


data_regression = data[[regression_target] + regression_features]


data_regression.dropna(inplace=True)

#data_regression = pd.get_dummies(data_regression, columns=['Post Weekday'], drop_first=True)
#step 9
# Split data into input features and target variable for regression
X_regression = data_regression.drop(columns=[regression_target])
y_regression = data_regression[regression_target]

# Split data into train and test sets for regression
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)


regression_model = LinearRegression()
regression_model.fit(X_train_regression, y_train_regression)


y_pred_regression = regression_model.predict(X_test_regression)


mse_regression = mean_squared_error(y_test_regression, y_pred_regression)
print("Mean Squared Error (Regression):", mse_regression)


#For the regression task, the target variable selected is 'Lifetime Post Total Reach'. 
#This variable represents the total number of unique users who have seen a specific post at least once.

# the features is ['Page total likes', 'Post Weekday', 'Post Hour', 'Lifetime Post Total Impressions',
                      #'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                      #'Lifetime Post Impressions by people who have liked your Page',
                      #'Lifetime Post reach by people who like your Page',
                      #'Lifetime People who have liked your Page and engaged with your post',
                      #'comment', 'like', 'share', 'Total Interactions']


#Explanation:

#These features were chosen because they are likely to influence the reach and engagement of our data  post. 
#Factors such as the number of likes, shares, and comments directly reflect user engagement, 
# while the time and day of the post might affect how many users see and interact with it.
# Additionally, metrics related to impressions and reach provide insights into the visibility and audience reach of the post.

# %% Cell 15
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.svm import SVC
classification_target = 'Type'
classification_features = ['Page total likes', 'Post Hour', 'Lifetime Post Total Reach',
                            'Lifetime Post Total Impressions', 'Lifetime Engaged Users', 'Lifetime Post Consumers',
                            'Lifetime Post Consumptions', 'Lifetime Post Impressions by people who have liked your Page',
                            'Lifetime Post reach by people who like your Page',
                            'Lifetime People who have liked your Page and engaged with your post',
                            'comment', 'like', 'share', 'Total Interactions']

data_classification = data[[classification_target] + classification_features]

data_classification.dropna(inplace=True)

label_encoder = LabelEncoder()
data_classification[classification_target] = label_encoder.fit_transform(data_classification[classification_target])

#data_classification = pd.get_dummies(data_classification, columns=['Post Weekday'], drop_first=True)
#step 9
# Split data into input features and target variable for classification
X_classification = data_classification.drop(columns=[classification_target])
y_classification = data_classification[classification_target]

# Split data into train and test sets for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_classification, y_train_classification)
rf_predictions = rf_classifier.predict(X_test_classification)
rf_accuracy = accuracy_score(y_test_classification, rf_predictions)
print("RandomForestClassifier Accuracy:", rf_accuracy)
print("Classification Report:")
print(classification_report(y_test_classification, rf_predictions))


svc_classifier = SVC(random_state=42)
svc_classifier.fit(X_train_classification, y_train_classification)
svc_predictions = svc_classifier.predict(X_test_classification)
svc_accuracy = accuracy_score(y_test_classification, svc_predictions)
print("\nSupport Vector Classifier (SVC) Accuracy:", svc_accuracy)
print("Classification Report:")
print(classification_report(y_test_classification, svc_predictions))

#Target Variable:

#For the classification task, the target variable chosen is 'Type'. 
#This variable represents the type of post, which could be a photo, status, link, or video.

#features is ['Page total likes', 'Post Weekday', 'Post Hour', 'Lifetime Post Total Reach',
                            #'Lifetime Post Total Impressions', 'Lifetime Engaged Users', 'Lifetime Post Consumers',
                            #'Lifetime Post Consumptions', 'Lifetime Post Impressions by people who have liked your Page',
                            #'Lifetime Post reach by people who like your Page',
                            #'Lifetime People who have liked your Page and engaged with your post',
                            #'comment', 'like', 'share', 'Total Interactions']
                    
#Explanation:

#These features were chosen because they provide valuable information about the characteristics of the post and how users interact with it
# which could influence the type of post it is.
#For example, certain types of posts might be more likely to receive a higher number of likes or shares, or they might be more popular on specific days or times.


# %% Cell 16
# We processed a data_15 post dataset, conducted exploratory data analysis, and applied machine learning models for both regression and classification tasks in this thorough investigation. Loading the dataset and fixing any missing values were the first stages. 
# To preserve the dataset's integrity, we substituted the mode for missing non-numeric values and the median for missing numeric values. To maintain uniformity in the data, duplicate entries were eliminated.
# In order to prepare categorical data for modeling, the categorical variables "Type" and "Category" were next encoded using Label Encoding, while "Post Weekday" was one-hot encoded. 
# To put the numerical features on a comparable scale—a critical step for algorithms that are sensitive to feature magnitudes—we normalized them using MinMaxScaler.

# We created box plots and histograms for exploratory data analysis in order to comprehend the distribution and spot possible outliers in the numerical features.
# A study examining the association between 'like' and'share' yielded a scatter plot that indicated a positive correlation.
# Finding duplicate variables was made easier with the use of a heatmap of the correlation matrix, which revealed information about the correlations between the characteristics. 
# Understanding the overall data structure was made easier by the pair plot, which provided a thorough perspective of the interactions among certain attributes

# Clustering techniques, including K-Means and Agglomerative Clustering, were applied to group similar data points. The Elbow method was used to determine the optimal number of clusters, which was found to be three. 
# The clusters formed by both algorithms were visualized, showing distinct groupings based on features like 'Lifetime Post Total Reach' and 'Lifetime Engaged Users'.

# We used data like "Page total likes," "Post Weekday," and other engagement metrics to estimate "Lifetime Post Total Reach" for the regression challenge.
# The accuracy of the model's predictions was indicated by the Mean Squared Error (MSE) that the Linear Regression model produced. 
# Our objective was to anticipate the post's 'Type' for classification. Support Vector Classifier (SVC) and RandomForestClassifier were used; RandomForestClassifier performed better in terms of accuracy.
# Accuracy ratings and classification reports, which emphasized the models' functionality and prediction quality, were used to assess both models.

# To sum up, the preprocessing procedures—which included encoding categorical variables and resolving missing values—were crucial in getting the data ready for analysis. 
# The links between features and the distribution of data were clearly shown by the visuals. The machine learning models successfully completed regression and classification tests, and clustering showed underlying patterns.
# The outcomes give support for the preprocessing and modeling decisions made because they meaningfully predicted the structure of the data and captured it well.
# Combining predictive modeling, clustering, and exploratory analysis provides a strong framework for deciphering and using social media data for strategic insights.
