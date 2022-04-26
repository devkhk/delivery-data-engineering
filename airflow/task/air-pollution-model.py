#
# Airflow task
# 저장되어있는 tarin, test 데이터를 불러온다.
# 미리 구한 하이퍼 파라미터 값을 불러온다.
# 하이퍼 파라미터값을 대입해 모델을 학습하고 모델을 저장한다.
#

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pandas as pd


MAX_MEMORY = "5g"
spark = SparkSession.builder.appName("air-pollution-degree-analysis")\
                            .config("spark.executor.memory", MAX_MEMORY)\
                            .config("spark.driver.memory", MAX_MEMORY)\
                            .getOrCreate()

data_dir = "/Users/devkhk/Documents/public-data-engineering/airflow/data/"

# 필요한 데이터, 하이퍼 파라미터를 불러온다.
train_df = spark.read.parquet(f"{data_dir}/train/")
test_df = spark.read.parquet(f"{data_dir}/test/")
hyper_df = pd.read_csv(f"{data_dir}hyperparameter.csv")

alpha = hyper_df.iloc[0]['alpha']
reg_param = hyper_df.iloc[0]['reg_param']

# pipeline stages 설계
stages = []

cat_features = [
    "region_code",
    "hour",
]

std_features = [
    "sulfur_diox",
    "fine_dust",
    "ozone",
    "nitrogen_diox",
    "carbon_monox",
]

for c in cat_features:
    indexer = StringIndexer(inputCol=c , outputCol= c + "_idx").setHandleInvalid("skip")
    onehot = OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol= c+ "_one")
    stages += [indexer, onehot]
for s in std_features:
    vassembler = VectorAssembler(inputCols=[s], outputCol=s + "_vc")
    stdscaler = StandardScaler(inputCol=vassembler.getOutputCol(), outputCol=s + "_std")
    stages += [vassembler, stdscaler]

# vector된 데이터들을 하나로 모으는 assembler
assembler_list = [c + "_one" for c in cat_features ] + [s + "_std" for s in std_features]
assembler = VectorAssembler(inputCols=assembler_list, outputCol="features")
stages += [assembler]

lr = LinearRegression(
    maxIter=100,
    labelCol="u_fine_dust",
    solver="normal",
    regParam=reg_param,
    elasticNetParam=alpha,
)

pipeline = Pipeline(stages=stages)
fitted_pipeline = pipeline.fit(train_df)

vtrain_df = fitted_pipeline.transform(train_df)
vtest_df = fitted_pipeline.transform(test_df)

# 모델 학습
model = lr.fit(vtrain_df)

prediction = model.transform(vtest_df)

prediction.cache()
prediction.select("fine_dust","u_fine_dust", "prediction").show()

# 모델 저장
model_dir = "/Users/devkhk/Documents/public-data-engineering/airflow/data/model/"
model.write().overwrite().save(model_dir)