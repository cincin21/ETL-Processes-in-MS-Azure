# Imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, DoubleType, IntegerType
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.tuning import TrainValidationSplitModel


# Create SparkSession
spark = SparkSession.builder \
    .appName('DataFrame') \
    .master('local[*]') \
    .getOrCreate()

spark.conf.set("spark.sql.shuffle.partitions", 5)
spark.conf.set("spark.driver.memory", "2g")
spark.conf.set("spark.executor.memory", "8g")

# Load data
path = "D:/Edu/big_data_project/data/raw/winequality-red.csv"
data = spark.read.load(path=path, format="csv", sep=";", inferSchema="true", header="true")
data.show(n=5, truncate=False)

# Select the columns to be used as the features (all except `quality`)
data= data.withColumnRenamed("quality", "label")
featureColumns = [c for c in data.columns if c != 'label']

# Create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")

# Transform the original data
data = assembler.transform(data)
data.show(5)
#%%
# Split the input data into training and test DataFrames with 80% to 20% weights
train_data, test_data = data.randomSplit([0.80, 0.20], seed=12345)

# Load Random Forest Classifier
rfc = RandomForestClassifier(featuresCol="features", labelCol="label", )

# Build ParamGrid
paramGrid = ParamGridBuilder()\
    .addGrid(rfc.impurity, ["entropy", "gini"]) \
    .addGrid(rfc.numTrees, [10, 25, 50, 75, 100, 150, 200, 250]) \
    .addGrid(rfc.maxDepth, [2,3,5, 10, 15, 20, 25, 30]) \
    .addGrid(rfc.maxBins, [5, 10, 20, 25, 30, 50, 75]) \
    .build()

tvs = TrainValidationSplit(estimator=rfc,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.80)
rfc_model = tvs.fit(train_data)

#Show best model params
bestModel = rfc_model.bestModel
print(f"Best Param (impurity): {bestModel._java_obj.getImpurity()}")
print(f"Best Param (numTrees): {bestModel._java_obj.getNumTrees()}")
print(f"Best Param (maxDepth): {bestModel._java_obj.getMaxDepth()}")
print(f"Best Param (maxBins): {bestModel._java_obj.getMaxBins()}")

# Define the random forest estimator
rfc_preds = rfc_model.transform(test_data).select("features", "label", "prediction")
rfc_preds.show()

# Calculate model accuracy
accuracy_eval = MulticlassClassificationEvaluator(metricName = 'accuracy', labelCol='label')
print(f"RandomForestClassifier accuracy: {accuracy_eval.evaluate(rfc_preds)}")

# Save model
rfc_model.save("D:/Edu/big_data_project/data/temp/rfc_model")

#%% Load model
loaded_model = TrainValidationSplitModel.load("D:/Edu/big_data_project/models/rfc_model")
loaded_preds = loaded_model.transform(test_data).select("features", "label", "prediction")
loaded_preds.show()

# Calculate Loaded model accuracy
accuracy_eval = MulticlassClassificationEvaluator(metricName = 'accuracy', labelCol='label')
print(f"RandomForestClassifier accuracy: {accuracy_eval.evaluate(loaded_preds)}")

#%% RMSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

predictions = loaded_preds.toPandas()
n = len(predictions)
rmse = np.linalg.norm(predictions["prediction"] - predictions["label"]) / np.sqrt(n)

predictions["difference"] = predictions["prediction"] - predictions["label"]

predictions["difference"].nunique()

sns.catplot(x="difference", kind="count", color="#6b007b", data=predictions)
plt.title("Difference between predicted quality label \n and original quality label.")
plt.tight_layout()
plt.savefig(r"D:\Edu\big_data_project\reports\figures\Difference_model_quality.png")
plt.show()