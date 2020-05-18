# Databricks notebook source
# MAGIC %md ## PROJECT GOAL:
# MAGIC Outbrain is a web advertising platform that displays advertisement boxes of links to pages within websites. It displays links to the sites' pages in addition to sponsored content, generating revenue from the latter.
# MAGIC In our project we are predicting whether the particular ad_id will be clicked or not.
# MAGIC 
# MAGIC This python files predict ad using random forest classifier algorithm and measure Accuracy and AUC.

# COMMAND ----------

# MAGIC %md ## Creating a Classification Model using Logistic Regression Algorithm
# MAGIC 
# MAGIC In this file, we will implement a classification model using *Logistic Regression that uses features of a Event,Document and Train dataframe to **predict the whether the ad is clicked or not** 
# MAGIC 
# MAGIC You should follow the steps below to build, train and test the model from the source data:
# MAGIC 
# MAGIC 1. Build a schema of a source data for its Data Frame
# MAGIC 2. Load the Source Data to the schema
# MAGIC 3. Prepare the data with the features (input columns, output column as label)
# MAGIC 4. Split the data using data.randomSplit(): Training and Testing
# MAGIC 5. Transform the columns to a vector using VectorAssembler
# MAGIC 6. set features and label from the vector
# MAGIC 7. Build a ***Logistic Regression Algorithm*** Model with the label and features
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
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier



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

# MAGIC %md #####Reading all necessary csv file 

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    event = spark.read.csv('eventd.csv', inferSchema=True, header=True)
else:
    event = spark.sql("SELECT * FROM eventd_csv")

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    train = spark.read.csv('traind.csv', inferSchema=True, header=True)
else:
    train = spark.sql("SELECT * FROM traind_csv")

# COMMAND ----------

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

data = data2.select("display_id", "document_id", "platform", "ad_id", "campaign_id","advertiser_id",col("clicked").alias("label"))

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

from pyspark.ml.classification import RandomForestClassifier
assembler = VectorAssembler(inputCols = ["display_id", "document_id", "platform", "ad_id","campaign_id","advertiser_id"],outputCol="normfeatures")
#assembler = VectorAssembler(inputCols = ["clicked"],outputCol="label")
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="nfeatures")
featVect = VectorAssembler(inputCols=["nfeatures"], outputCol="features")
#rf = RandomForestClassifier(labelCol="label", featuresCol="features",impurity="gini",featureSubsetStrategy="auto",numTrees=10,maxDepth=32,maxBins=128,seed=1234)
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)
pipeline = Pipeline(stages=[assembler,minMax,featVect,lr])

# COMMAND ----------

# MAGIC %md ### Tune Parameters
# MAGIC You can tune parameters to find the best model for your data. To do this you can use the  **CrossValidator** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets, in order to find the best performing parameters. Note that this can take a long time to run because every parameter combination is tried multiple times.

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
# TODO: K = 2, you may test it with 5, 10
# K=2, 5, 10: Root Mean Square Error (RMSE): 13.2
cv = CrossValidator(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, numFolds=10)

model = cv.fit(train)


# COMMAND ----------

# MAGIC %md  ### Test the Recommender
# MAGIC The model produced by the pipeline is a transformed that will apply to all stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the test DataFrame using the pipeline to generate label prediction

# COMMAND ----------

# piplineModel with train data set applies test data set and generate predictions
prediction = model.transform(test)
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
print("Accuracy of Logistic Regression is: ",accuracy)

# COMMAND ----------

# MAGIC %md Following Code will give total Error in project

# COMMAND ----------

print("Test Error = %g" % (1.0 - accuracy))

# COMMAND ----------

# MAGIC %md #### Logistic Regression Evaluator
# MAGIC Calculate AUC using Logistic Regression Evaluator.

# COMMAND ----------

rf_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
rf_auc = rf_evaluator.evaluate(prediction)
print("AUC for Logistic Regression is= ", rf_auc)

# COMMAND ----------

# MAGIC %md #### Logistic Regression Confusion matrix
# MAGIC Calculate confusion matrix and measure precision,Recall

# COMMAND ----------

tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp ),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()

