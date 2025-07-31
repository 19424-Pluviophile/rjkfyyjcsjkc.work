import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
import requests


# 加载训练历史记录
def load_train_history(task_name):
    with open(f"train_history_{task_name}.pkl", "rb") as f:
        return pickle.load(f)


# 绘制训练曲线
def plot_training_curves(task_name):
    history = load_train_history(task_name)
    plt.figure(figsize=(12, 4))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.title(f"{task_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss")
    plt.title(f"{task_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{task_name}_training_curves.svg")
    plt.close()


# 模型评估
def evaluate_model(task_name, data_path):
    # 加载验证集
    spark = SparkSession.builder.getOrCreate()
    dev_df = spark.read.parquet(f"{data_path}/processed/{task_name}/dev")

    # 加载模型
    model = load_model(f"bert_{task_name}_best.h5", custom_objects={"TFBertModel": TFBertModel})

    # 转换为评估格式
    inputs = {
        "input_ids": np.array(dev_df.select("input_ids").rdd.map(lambda x: x[0]).collect()),
        "attention_mask": np.array(dev_df.select("attention_mask").rdd.map(lambda x: x[0]).collect()),
        "token_type_ids": np.array(dev_df.select("token_type_ids").rdd.map(lambda x: x[0]).collect())
    }
    labels = np.array(dev_df.select("label").rdd.map(lambda x: x[0]).collect())

    # 评估
    loss, accuracy = model.evaluate(inputs, labels, verbose=0)

    # 计算F1值（针对MRPC）
    if task_name == "MRPC":
        from sklearn.metrics import f1_score
        preds = np.argmax(model.predict(inputs), axis=1)
        f1 = f1_score(labels, preds)
        return {"loss": loss, "accuracy": accuracy, "f1": f1}
    return {"loss": loss, "accuracy": accuracy}


# 生成GLUE官网提交格式
def generate_glue_submission(task_name, data_path):
    # 加载测试集
    spark = SparkSession.builder.getOrCreate()
    test_df = spark.read.parquet(f"{data_path}/processed/{task_name}/test")

    # 加载模型并预测
    model = load_model(f"bert_{task_name}_best.h5", custom_objects={"TFBertModel": TFBertModel})
    inputs = {
        "input_ids": np.array(test_df.select("input_ids").rdd.map(lambda x: x[0]).collect()),
        "attention_mask": np.array(test_df.select("attention_mask").rdd.map(lambda x: x[0]).collect()),
        "token_type_ids": np.array(test_df.select("token_type_ids").rdd.map(lambda x: x[0]).collect())
    }
    preds = np.argmax(model.predict(inputs), axis=1)

    # 生成提交文件（按GLUE要求格式）
    submission = {
        "id": test_df.select("idx").rdd.map(lambda x: x[0]).collect(),
        "prediction": preds.tolist()
    }
    with open(f"{task_name}_submission.json", "w") as f:
        json.dump(submission, f)


# 主函数
if __name__ == "__main__":
    data_path = "/root/glue" if os.path.exists(
        "/root/glue") else "C:\\Users\\86198\\Desktop\\学习\\study\\课程学习\\计算机\\软件开发应用基础实践\\glue"

    # 评估SST-2
    sst2_metrics = evaluate_model("SST-2", data_path)
    print(f"SST-2 Evaluation: Accuracy={sst2_metrics['accuracy']:.4f}, Loss={sst2_metrics['loss']:.4f}")

    # 评估MRPC
    mrpc_metrics = evaluate_model("MRPC", data_path)
    print(
        f"MRPC Evaluation: Accuracy={mrpc_metrics['accuracy']:.4f}, F1={mrpc_metrics['f1']:.4f}, Loss={mrpc_metrics['loss']:.4f}")

    # 绘制训练曲线
    plot_training_curves("SST-2")
    plot_training_curves("MRPC")

    # 生成提交文件
    generate_glue_submission("SST-2", data_path)
    generate_glue_submission("MRPC", data_path)
