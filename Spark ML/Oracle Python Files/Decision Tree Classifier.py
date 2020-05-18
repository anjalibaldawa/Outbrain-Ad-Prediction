# Databricks notebook source
# MAGIC %md ## PROJECT GOAL:
# MAGIC Outbrain is a web advertising platform that displays advertisement boxes of links to pages within websites. It displays links to the sites' pages in addition to sponsored content, generating revenue from the latter.
# MAGIC In our project we are predicting whether the particular ad_id will be clicked or not.
# MAGIC 
# MAGIC This python files predict ad using random forest classifier algorithm and measure Accuracy and AUC.

# COMMAND ----------

# MAGIC %md ## Creating a Classification Model using Decision Tree Classifier Algorithm
# MAGIC 
# MAGIC In this file, we will implement a classification model using *decision tree Classifier* that uses features of a Event,Document and Train dataframe to **predict the whether the ad is clicked or not** 
# MAGIC 
# MAGIC You should follow the steps below to build, train and test the model from the source data:
# MAGIC 
# MAGIC 1. Build a schema of a source data for its Data Frame
# MAGIC 2. Load the Source Data to the schema
# MAGIC 3. Prepare the data with the features (input columns, output column as label)
# MAGIC 4. Split the data using data.randomSplit(): Training and Testing
# MAGIC 5. Transform the columns to a vector using VectorAssembler
# MAGIC 6. set features and label from the vector
# MAGIC 7. Build a ***Decision Tree Classifier Algorithm*** Model with the label and features
# MAGIC 8. Train the model
# MAGIC 9. Prepare the testing Data Frame with features and label from the vector; Rename label to trueLabel
# MAGIC 10. Predict and test the testing Data Frame using the model trained at the step 8
# MAGIC 11. Compare the predicted result and trueLabel
# MAGIC 
# MAGIC 
# MAGIC ### Import Spark SQL and Spark ML Libraries
# MAGIC 
# MAGIC First, import the libraries you will need:

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.mllib.tree import RandomForest
from pyspark.ml import Pipeline

from pyspark.ml.classification import DecisionTreeClassifier

from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import MultilayerPerceptronClassifier



# COMMAND ----------

# MAGIC %md #####The below command will start spark session when we run our file in oracle BDE. In Databricks keep this cell as False by default. But when you run file in Oracle BDE make it True.

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = False
if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %md ### Load Source Data
# MAGIC The data for this project is provided as a CSV file containing details of advertisement. The data includes specific characteristics (or *features*) for each ad, as well as a *label* column indicating whether the ad was clicked or not.
# MAGIC 
# MAGIC You will load this data into a DataFrame and display it.

# COMMAND ----------

eventSchema = StructType([
  StructField("display_id", IntegerType(), False),
  StructField("uuid", StringType(), False),
  StructField("document_id", IntegerType(), False),
  StructField("timestamp", IntegerType(), False),
  StructField("platform", IntegerType(), False),
  StructField("geo_location", StringType(), False)
])

# COMMAND ----------

# MAGIC %md ### Prepare the Data
# MAGIC Most modeling begins with exhaustive exploration and preparation of the data. In this example, you will simply select a subset of columns to use as *features* as well as the **ArrDelay** column, which will be the *label* your model will predict.

# COMMAND ----------

# MAGIC %md #####Reading all necessary csv file

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    event = spark.read.csv('events.csv', inferSchema=True, header=True)
else:
    event = spark.sql("SELECT * FROM eventd_csv")

# COMMAND ----------

#create train schema
trainSchema = StructType([
  StructField("display_id", DoubleType(), False),
  StructField("DayOfWeek", DoubleType(), False),
  StructField("clicked", IntegerType(), False)
])

# COMMAND ----------

#Read traind.csv file
if IS_SPARK_SUBMIT_CLI:
    train = spark.read.csv('trains.csv', inferSchema=True, header=True)
else:
    train = spark.sql("SELECT * FROM traind_csv")

# COMMAND ----------

# create Promoted_Content Schema
promoted_contentSchema = StructType([
  StructField("ad_id", IntegerType(), False),
  StructField("document_id", IntegerType(), False),
  StructField("campaign", IntegerType(), False),
  StructField("advertiser_id", IntegerType(), False)
])

# COMMAND ----------

# Read Promoted_content.csv file
if IS_SPARK_SUBMIT_CLI:
    promoted_content = spark.read.csv('promoted_content.csv', inferSchema=True, header=True)
else:
    promoted_content = spark.sql("SELECT * FROM promoted_content_csv")

# COMMAND ----------

# MAGIC %md Below command will join event and train file using disply_id as primary key. It will also drop display_id from event table and ad_id from train table as it create ambiguity error. Then it will assign result to new dataframe data1.

# COMMAND ----------

data1=event.join(train,event.display_id==train.display_id).drop(event.display_id).drop(train.ad_id)

# COMMAND ----------

# MAGIC %md Following cell will join two dataframe data1 (which created from joining of event and train table) with Promoted content dataframe using document_id as primary key. Then it will assign result to new dataframe data2.

# COMMAND ----------

data2=data1.join(promoted_content,data1.document_id==promoted_content.document_id).drop(data1.document_id)

# COMMAND ----------

# MAGIC %md Following table select disply_id,document_id,platform,ad_id,camaign_id,advertiser_id which are features which help us to train our dataset and clicked column which is label column. Selected column will assign to new dataframe called data.

# COMMAND ----------

data =  data2.select(col("display_id").cast(DoubleType()), col("document_id").cast(DoubleType()), col("platform").cast(DoubleType()), col("ad_id").cast(DoubleType()), col("campaign_id").cast(DoubleType()),col("advertiser_id").cast(DoubleType()),col("clicked").alias("label"))

# COMMAND ----------

# MAGIC %md ### Split the Data
# MAGIC It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])     
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

# MAGIC %md ### Prepare the Training Data create Pipeline
# MAGIC To train the classification model, you need a training data set that includes a vector of numeric features, and a label column. In this exercise, you will use the **VectorAssembler** class to transform the feature columns into a vector,Normlize data using Minmax.

# COMMAND ----------

# Import RandomClassifier Algorithm 
from pyspark.ml.classification import RandomForestClassifier

# Convert following feature column in one vector for train data
assembler = VectorAssembler(inputCols = ["display_id", "document_id", "platform", "ad_id","campaign_id","advertiser_id"],outputCol="normfeatures")
#assembler = VectorAssembler(inputCols = ["clicked"],outputCol="label")

#Normlize feature data 
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="nfeatures")

#Convert Normlize feature data to vector
featVect = VectorAssembler(inputCols=["nfeatures"], outputCol="features")

#following Random forest algorithm train the classifiction model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",impurity='gini',maxDepth=30,maxBins=128,seed=1234)
# Following command will create pipeline with different stages
pipeline = Pipeline(stages=[assembler,minMax,featVect,dt])

# COMMAND ----------

# MAGIC %md The pipeline itself is an estimator, and so it has a fit method that you can call to run the pipeline on a specified DataFrame. In this case, you will run the pipeline on the training data to train a model.

# COMMAND ----------

piplineModel = pipeline.fit(train)
print("Pipeline complete!")


# COMMAND ----------

# MAGIC %md ### Test the Recommender 
# MAGIC The model produced by the pipeline is a transformed that will apply to all stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the test DataFrame using the pipeline to generate label prediction

# COMMAND ----------

# piplineModel with train data set applies test data set and generate predictions
prediction = piplineModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Following cell measure accuracy of Algorithm using MultiClassificationEvaluater and evaluate using predicted data.

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluation = MulticlassClassificationEvaluator(
    labelCol="trueLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluation.evaluate(prediction)
print("Accuracy for Decision Tree classifier is=",accuracy)

# COMMAND ----------

# MAGIC %md Following Code will give total Error in project

# COMMAND ----------

print("Test Error = %g" % (1.0 - accuracy))

# COMMAND ----------

# MAGIC %md #### Decision Tree Evaluator
# MAGIC Calculate AUC using Decision Tree Evaluator.

# COMMAND ----------

rf_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
rf_auc = rf_evaluator.evaluate(prediction)
print("AUC for Decision Tree is= ", rf_auc)

# COMMAND ----------

# MAGIC %md #### Decision Tree Classifier Confusion matrix
# MAGIC Calculate confusion matrix and measure precision,Recall

# COMMAND ----------

# Only for Classification Logistic Regression not for Linear Regression

tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()

