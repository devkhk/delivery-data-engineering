#
# Airflow task
# 기존 데이터에서 하이퍼 파라미터를 구한다.
# 하이퍼 파라미터 값을 csv파일로 저장한다. (hyperparameter.csv)
#


from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler, StandardScaler

import pandas as pd


MAX_MEMORY = "5g"
spark = SparkSession.builder.appName("air-pollution-degree-analysis")\
                            .config("spark.executor.memory", MAX_MEMORY)\
                            .config("spark.driver.memory", MAX_MEMORY)\
                            .getOrCreate()

data_dir = "/Users/devkhk/Documents/public-data-engineering/airflow/data/"

train_df = spark.read.parquet(f"{data_dir}/train/")
toy_df = train_df.sample(False, fraction=0.1, seed=1)

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

cv_lr = LinearRegression(
    maxIter=30,
    labelCol="u_fine_dust",
    solver="normal",
)
cv_stages = stages + [cv_lr]

cv_pipeline = Pipeline(stages=cv_stages)

param_grid = ParamGridBuilder()\
                .addGrid(cv_lr.regParam, [0.01, 0.02, 0.03, 0.04, 0.05])\
                .addGrid(cv_lr.elasticNetParam, [0.1, 0.2, 0.3, 0.4, 0.05])\
                .build()

cv = CrossValidator(
        estimator=cv_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=RegressionEvaluator(labelCol="u_fine_dust"),
        numFolds=5
)

cv_model = cv.fit(toy_df)

alpha = cv_model.bestModel.stages[-1]._java_obj.getElasticNetParam()
reg_param = cv_model.bestModel.stages[-1]._java_obj.getRegParam()

# 불러와서 사용하기 편하게 벡터로 저장
hyper_param = {
    'alpha':[alpha],
    'reg_param':[reg_param]
}

hyper_df = pd.DataFrame(hyper_param).to_csv(f"{data_dir}hyperparameter.csv")
print(hyper_df)