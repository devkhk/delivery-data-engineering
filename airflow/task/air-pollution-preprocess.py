#
# Airflow task
# 대기 오염 분석 데이터를 가공해 parquet로 저장한다.
#


from pyspark.sql import SparkSession

MAX_MEMORY = "5g"
spark = SparkSession.builder.appName("air-pollution-degree-analysis")\
                            .config("spark.executor.memory", MAX_MEMORY)\
                            .config("spark.driver.memory", MAX_MEMORY)\
                            .getOrCreate()

air_pollution_dir = "/Users/devkhk/Documents/public-data-engineering/data/air_pollution_degree/"
air_pollution_df = spark.read.csv(f"file:///{air_pollution_dir}air-pollution-degree.csv", encoding="euc-kr", header=True, inferSchema=True)\
                            .toDF("city", "city2", "region", "region2", "region_code", "measure_date", "sulfur_diox", "fine_dust", "ozone", "nitrogen_diox", "carbon_monox","u_fine_dust")
air_pollution_df.createOrReplaceTempView("origin")

query = """
SELECT
    city,
    region,
    region_code,
    TO_DATE(measure_date) as date,
    HOUR(measure_date) as hour,
    sulfur_diox,
    fine_dust,
    ozone,
    nitrogen_diox,
    carbon_monox,
    u_fine_dust
FROM origin
"""
origin_df = spark.sql(query)
origin_df.createOrReplaceTempView("origin_preprocess")

query = """
SELECT
    *
FROM
    origin_preprocess
WHERE
        sulfur_diox > 0
    and ozone > 0
    and nitrogen_diox > 0
    and carbon_monox > 0
    and carbon_monox < 200
    and fine_dust > 0
    and u_fine_dust > 0

"""

preprocessed_df = spark.sql(query)
preprocessed_df = preprocessed_df.na.drop("any")

train_df, test_df =  preprocessed_df.randomSplit([0.8, 0.2], seed=1)

data_dir = "/Users/devkhk/Documents/public-data-engineering/airflow/data/"
train_df.write.format("parquet").mode("overwrite").save(path=f"{data_dir}/train/")
test_df.write.format("parquet").mode("overwrite").save(path=f"{data_dir}/test/")