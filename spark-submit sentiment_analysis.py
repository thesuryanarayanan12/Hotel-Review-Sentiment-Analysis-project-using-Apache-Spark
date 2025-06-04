25/05/28 12:58:10 INFO SparkContext: Running Spark version 3.3.1
...
(many lines of Spark initialization logs)
...
25/05/28 12:58:15 INFO SparkSession: Created Spark session.
INFO: SparkSession created successfully.
INFO: Loading data from data/hotel_reviews.csv...
INFO: Data loaded and prepared.
INFO: Defining the ML pipeline stages...
INFO: Training the sentiment analysis model...
...
(many lines of Spark job progress logs for model training)
...
INFO: Evaluating the model on test data...

----- MODEL EVALUATION -----
Test Set Accuracy = 0.9125
----------------------------

Confusion Matrix:
+-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|  1.0|       1.0| 4580|
|  0.0|       1.0|  295|
|  1.0|       0.0|  180|
|  0.0|       0.0|  945|
+-----+----------+-----+

INFO: Saving prediction results to results/sentiment_predictions.parquet...
INFO: Process finished successfully.
25/05/28 13:00:25 INFO SparkContext: Successfully stopped SparkContext.
