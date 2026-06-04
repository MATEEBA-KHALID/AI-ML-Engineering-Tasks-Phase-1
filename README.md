# AI-ML-Engineering-Tasks-Phase-1
Intership tasks done at DevelopersHub Corporation


# Task 1: Exploring and Visualizing the Iris Dataset

## Objective

To learn how to load, inspect, analyze, and visualize a dataset using Python data science libraries.

## Dataset Used

* Iris Dataset
* Loaded using Seaborn (`sns.load_dataset('iris')`)
* Contains 150 flower samples from three iris species:

  * Setosa
  * Versicolor
  * Virginica

## Models Applied

No machine learning model was used in this task. The focus was on Exploratory Data Analysis (EDA).

## Techniques Used

* Pandas for data loading and inspection
* Descriptive statistics using `.info()` and `.describe()`
* Scatter plots
* Histograms
* Box plots
* Matplotlib and Seaborn visualizations

## Key Results and Findings

* The dataset contains 150 records and 5 attributes.
* Petal measurements clearly separate different flower species.
* Histograms showed the distribution of numerical features.
* Box plots helped identify potential outliers.
* Scatter plots revealed strong relationships between petal dimensions.


# Task 2: Predict Future Stock Prices

## Objective

To predict the next day's stock closing price using historical stock market data.

## Dataset Used

* Historical stock data retrieved using the yfinance library.
* Example stocks:

  * Apple (AAPL)
  * Tesla (TSLA)

## Features Used

* Open Price
* High Price
* Low Price
* Volume

## Target Variable

* Next Day Closing Price

## Models Applied

* Linear Regression
* Random Forest Regressor (alternative)

## Key Results and Findings

* Historical market indicators were used to estimate future closing prices.
* Linear Regression provided a simple baseline prediction model.
* Random Forest improved prediction performance by capturing nonlinear relationships.
* Actual and predicted prices were compared using visualization graphs.
* Model performance was evaluated using:

  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
  * R² Score


# Task 3: Heart Disease Prediction

## Objective

To predict whether a patient is at risk of heart disease using medical data.

## Dataset Used

* Heart Disease UCI Dataset
* Downloaded from Kaggle

## Features Used

Common attributes include:

* Age
* Sex
* Chest Pain Type
* Cholesterol
* Blood Pressure
* Maximum Heart Rate
* Exercise-Induced Angina
* Other medical measurements

## Target Variable

* 0 = No Heart Disease
* 1 = Heart Disease Present

## Models Applied

* Logistic Regression
* Decision Tree (optional alternative)

## Key Results and Findings

* Data cleaning was performed to handle missing values.
* Exploratory Data Analysis identified trends among medical variables.
* Logistic Regression successfully classified patients into risk categories.
* Model evaluation included:

  * Accuracy Score
  * Confusion Matrix
  * ROC Curve
  * ROC-AUC Score
* Feature importance analysis highlighted the most influential health indicators affecting prediction.


# Task 4: General Health Query Chatbot

## Objective

To build a chatbot that answers general health-related questions using a Large Language Model (LLM).

## Dataset Used

No traditional dataset was required. The chatbot relies on a pretrained Large Language Model through an API.

## Models Applied

* OpenAI GPT-3.5 (via API)
* Mistral-7B-Instruct (alternative open-source model)

## Prompt Engineering Used

System Prompt:
"Act like a helpful medical assistant. Provide clear, friendly, and easy-to-understand health information while avoiding diagnoses and prescriptions."

## Safety Features

* Harmful medical advice prevention
* Restricted responses for dangerous requests
* Encourages professional healthcare consultation when necessary

## Key Results and Findings

* The chatbot successfully answered general health questions.
* Prompt engineering improved clarity and friendliness of responses.
* Safety filters reduced the risk of harmful outputs.
* The system demonstrated the practical use of LLMs for healthcare information assistance.


# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* yfinance
* OpenAI API
* Hugging Face Transformers

# Conclusion

These projects demonstrate fundamental concepts in data science, machine learning, and artificial intelligence, including data visualization, regression, classification, prompt engineering, and conversational AI. Together, they provide hands-on experience with real-world datasets and modern AI tools while emphasizing model evaluation, interpretability, and responsible AI usage.
