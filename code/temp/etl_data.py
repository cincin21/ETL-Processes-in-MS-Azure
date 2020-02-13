#%% Imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import TrainValidationSplitModel
from  pyspark.sql.functions import input_file_name


# Create SparkSession
spark = SparkSession.builder \
    .appName('DataFrame') \
    .master('local[*]') \
    .getOrCreate()

spark.conf.set("spark.sql.shuffle.partitions", 5)
spark.conf.set("spark.driver.memory", "2g")
spark.conf.set("spark.executor.memory", "8g")

# Load data
path = "D:/Edu/big_data_project/data/processed/*json"
data = spark.read.load(path=path, format="json", inferSchema="true")
data = data.withColumn("date", input_file_name().substr(-22,8))
data = data.withColumn("date", F.to_date(data["date"], "yyyyMMdd"))
data = data.withColumn("time", input_file_name().substr(-13,6))
data = data.withColumn("time", F.unix_timestamp("time", "HHmmss")) \
           .withColumn("time", F.from_unixtime("time", "HH:mm:ss"))
data = data.withColumn("location", input_file_name().substr(-6,1))

# Map locations column to proper names
location_dict = {
    "0":"Okanagan Valley",
    "1":"Bordeaux",
    "2":"New York",
    "3":"Mendoza",
    "4":"Willamette Valley",
    "5":"Tuscany",
    "6":"Cape Town",
    "7":"Napa & Sonoma",
    "8":"Barcelona",
    "9":"Yarra Valley",
    }
data = data.replace(to_replace=location_dict, subset=["location"])

# Select the columns to be used as the features
featureColumns = [c for c in ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                              "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

#Create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")

# Transform the original data
data = assembler.transform(data)

# Load model and predict wine quality
loaded_model = TrainValidationSplitModel.load("D:/Edu/big_data_project/data/temp/rfc_model")
loaded_preds = loaded_model.transform(data).select("features", "prediction")
loaded_preds.show()

# Create and sort final DataFrame
data = data.join(loaded_preds, on=["features"]).drop("features")
data = data.withColumnRenamed("prediction", "quality")
data = data.select("date", "time", "location", "quality", "fixed acidity", "volatile acidity", "citric acid",
                   "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                   "pH", "sulphates", "alcohol")
data = data.orderBy(data["date"], data["time"])
data.show()
