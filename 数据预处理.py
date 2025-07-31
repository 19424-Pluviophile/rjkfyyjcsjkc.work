from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, IntegerType, StringType
from pyspark.sql.types import StructType, StructField
from transformers import BertTokenizer
import os

# 初始化Spark会话
spark = SparkSession.builder \
    .appName("GLUE_Preprocessing") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# 定义数据集路径
# 本地路径
local_path = "C:\\Users\\86198\\Desktop\\学习\\study\\课程学习\\计算机\\软件开发应用基础实践\\glue"
# Linux路径
linux_path = "/root/glue"
data_path = local_path if os.path.exists(local_path) else linux_path

# 加载SST-2和MRPC数据集
def load_glue_dataset(task_name):
    """加载GLUE任务数据集"""
    train_df = spark.read.json(f"{data_path}/{task_name}/train.jsonl")
    dev_df = spark.read.json(f"{data_path}/{task_name}/dev.jsonl")
    test_df = spark.read.json(f"{data_path}/{task_name}/test.jsonl")
    return train_df, dev_df, test_df

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 定义分词UDF
def tokenize_text(sentence1, sentence2=None, max_len=128):
    """对文本进行分词，生成input_ids、attention_mask和token_type_ids"""
    if sentence2:
        inputs = tokenizer(
            sentence1, sentence2,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
    else:
        inputs = tokenizer(
            sentence1,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
    return {
        "input_ids": inputs["input_ids"].squeeze().tolist(),
        "attention_mask": inputs["attention_mask"].squeeze().tolist(),
        "token_type_ids": inputs["token_type_ids"].squeeze().tolist() if "token_type_ids" in inputs else []
    }

# 注册UDF
def register_tokenize_udf(task_name):
    if task_name in ["SST-2"]:  # 单句任务
        return udf(
            lambda x: tokenize_text(x),
            StructType([
                StructField("input_ids", ArrayType(IntegerType())),
                StructField("attention_mask", ArrayType(IntegerType())),
                StructField("token_type_ids", ArrayType(IntegerType()))
            ])
        )
    elif task_name in ["MRPC"]:  # 双句任务
        return udf(
            lambda x, y: tokenize_text(x, y),
            StructType([
                StructField("input_ids", ArrayType(IntegerType())),
                StructField("attention_mask", ArrayType(IntegerType())),
                StructField("token_type_ids", ArrayType(IntegerType()))
            ])
        )

# 处理SST-2数据集
sst2_train, sst2_dev, sst2_test = load_glue_dataset("SST-2")
sst2_tokenize_udf = register_tokenize_udf("SST-2")
sst2_train_processed = sst2_train.withColumn(
    "features", sst2_tokenize_udf(col("sentence"))
).select("features.input_ids", "features.attention_mask", "features.token_type_ids", col("label").cast(IntegerType()))

# 处理MRPC数据集
mrpc_train, mrpc_dev, mrpc_test = load_glue_dataset("MRPC")
mrpc_tokenize_udf = register_tokenize_udf("MRPC")
mrpc_train_processed = mrpc_train.withColumn(
    "features", mrpc_tokenize_udf(col("sentence1"), col("sentence2"))
).select("features.input_ids", "features.attention_mask", "features.token_type_ids", col("label").cast(IntegerType()))

# 保存处理后的数据为Parquet格式（分布式存储）
sst2_train_processed.write.mode("overwrite").parquet(f"{data_path}/processed/SST-2/train")
mrpc_train_processed.write.mode("overwrite").parquet(f"{data_path}/processed/MRPC/train")

spark.stop()
