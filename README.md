# Hotel Review Sentiment Classification Using Apache Spark

## Project Overview

This project aims to build a scalable machine learning pipeline to classify hotel reviews into **positive** or **negative** sentiments. Leveraging the distributed computing capabilities of Apache Spark, the pipeline efficiently processes large volumes of review data, performs feature engineering, and trains a classification model to predict sentiment labels with high accuracy.

The solution is designed to handle real-world datasets that may be large and unstructured, making Spark an ideal choice for distributed data processing and model training.

---

## Prerequisites

Before running the project, ensure the following are installed and configured:

* **Apache Spark** (version 3.0.0 or later)
* **Python** (version 3.7 or later)
* **PySpark** package installed (`pip install pyspark`)
* An input CSV file named `hotel_reviews.csv` located in the `data/` directory. This file must include at least these two columns:

  * `review_text`: the text content of each hotel review
  * `sentiment`: label for each review, where `1` represents a positive review and `0` represents a negative review

---

## Project Description

The pipeline consists of the following major steps:

1. **Data Loading and Preprocessing**

   * Load review data from CSV using Spark DataFrame APIs.
   * Clean and preprocess the text (e.g., removing punctuation, lowercasing).
   * Handle missing or corrupt data.

2. **Feature Engineering**

   * Tokenize the review text into words.
   * Remove stop words to reduce noise.
   * Apply TF-IDF vectorization to convert text into numerical features.

3. **Model Training and Evaluation**

   * Split the dataset into training and test sets.
   * Train a classification model using Spark MLlib, such as Logistic Regression or Random Forest.
   * Evaluate model performance with metrics like accuracy, precision, recall, and F1-score.

4. **Prediction**

   * Use the trained model to classify new reviews.
   * Save prediction results for further analysis.

---

## How to Run

1. Place your `hotel_reviews.csv` file inside the `data/` directory.
2. Run the main pipeline script (e.g., `spark_sentiment_analysis.py`):

   ```bash
   spark-submit spark_sentiment_analysis.py
   ```
3. The script will output logs describing each step and save the final predictions to `results/predictions.csv`.

---

## Sample Output

After running the pipeline, you should see output similar to the following:

```
Loading data from data/hotel_reviews.csv...
Cleaning and preprocessing text...
Tokenizing and removing stop words...
Applying TF-IDF vectorization...
Splitting dataset into training and testing sets...
Training Logistic Regression model...
Evaluating model performance...
Accuracy: 0.87
Precision: 0.85
Recall: 0.88
F1-score: 0.86
Saving prediction results to results/predictions.csv
Analysis complete.
```

Sample rows from `results/predictions.csv`:

| review\_text                           | actual\_sentiment | predicted\_sentiment |
| -------------------------------------- | ----------------- | -------------------- |
| "The room was spotless and very cozy." | 1                 | 1                    |
| "Worst experience ever, very noisy."   | 0                 | 0                    |
| "Staff were friendly and helpful."     | 1                 | 1                    |
| "Bathroom was dirty and smelled bad."  | 0                 | 0                    |

---

## Future Work

* Incorporate more advanced NLP techniques like word embeddings (Word2Vec, BERT).
* Perform hyperparameter tuning to improve model accuracy.
* Extend to multiclass classification for nuanced sentiment categories.
* Build a web app or API endpoint for real-time sentiment analysis.

---

## References

* [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
* [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
* [Sentiment Analysis with Spark](https://databricks.com/blog/2017/03/30/sentiment-analysis-with-apache-spark.html)
