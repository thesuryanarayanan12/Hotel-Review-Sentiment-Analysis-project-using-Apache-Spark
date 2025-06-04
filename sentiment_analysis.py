from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    """
    Main function to run the sentiment analysis pipeline.
    """
    # --- 1. INITIALIZE SPARK SESSION ---
    # Create a SparkSession, the entry point to Spark functionality.
    spark = SparkSession.builder \
        .appName("HotelReviewSentimentAnalysis") \
        .master("local[*]") \
        .getOrCreate()

    print("INFO: SparkSession created successfully.")

    # --- 2. LOAD DATA ---
    # Load the dataset. We assume the CSV has a header.
    # The key columns are 'review_text' and 'sentiment' (as 0.0 or 1.0)
    print("INFO: Loading data from data/hotel_reviews.csv...")
    df = spark.read.csv("data/hotel_reviews.csv", header=True, inferSchema=True)

    # Prepare the data by selecting relevant columns and renaming 'sentiment' to 'label' for MLlib
    data = df.select(col("review_text"), col("sentiment").alias("label")).na.drop()
    print("INFO: Data loaded and prepared.")

    # Split data into training (80%) and testing (20%) sets
    (train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

    # --- 3. BUILD THE ML PIPELINE ---
    # A pipeline chains multiple transformers and estimators together.
    print("INFO: Defining the ML pipeline stages...")

    # Stage 1: Tokenizer - Splits review text into individual words.
    tokenizer = Tokenizer(inputCol="review_text", outputCol="words")

    # Stage 2: StopWordsRemover - Removes common words (e.g., 'the', 'a', 'is').
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # Stage 3: HashingTF - Converts words to feature vectors using hashing.
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

    # Stage 4: IDF (Inverse Document Frequency) - Down-weights words that appear frequently.
    idf = IDF(inputCol="raw_features", outputCol="features")

    # Stage 5: Logistic Regression - The classification model.
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

    # Chain all stages into a Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])

    # --- 4. TRAIN AND EVALUATE THE MODEL ---
    print("INFO: Training the sentiment analysis model...")
    model = pipeline.fit(train_data)

    print("INFO: Evaluating the model on test data...")
    predictions = model.transform(test_data)

    # Use an evaluator to check the model's accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)

    print("\n----- MODEL EVALUATION -----")
    print(f"Test Set Accuracy = {accuracy:.4f}")
    print("----------------------------\n")

    # Show a confusion matrix
    print("Confusion Matrix:")
    predictions.groupBy("label", "prediction").count().show()


    # --- 5. SAVE RESULTS ---
    print("INFO: Saving prediction results to results/sentiment_predictions.parquet...")
    predictions.select("review_text", "label", "prediction").write.mode("overwrite").parquet("results/sentiment_predictions.parquet")

    print("INFO: Process finished successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
