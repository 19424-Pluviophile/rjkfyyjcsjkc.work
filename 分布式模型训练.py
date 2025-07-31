import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from transformers import TFBertModel
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor

# 初始化Horovod
hvd.init()

# 配置GPU（仅在分布式环境中由主进程执行）
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# 加载预处理数据（Spark DataFrame）
def load_processed_data(task_name, data_path):
    spark = SparkSession.builder.getOrCreate()
    return spark.read.parquet(f"{data_path}/processed/{task_name}/train")


# 构建BERT分类模型
def build_bert_model(num_labels=2):
    bert_base = TFBertModel.from_pretrained("bert-base-uncased")
    input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(128,), dtype=tf.int32, name="token_type_ids")

    bert_output = bert_base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )[1]  # 取[CLS] token的输出

    output = Dense(num_labels, activation="softmax")(bert_output)
    model = Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=output
    )

    # 配置Horovod优化器
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        experimental_run_tf_function=False
    )
    return model


# 训练函数
def train_task(task_name, data_path, epochs=3, batch_size=32):
    # 加载数据
    df = load_processed_data(task_name, data_path)
    # 转换为TensorFlow Dataset（分布式加载）
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": df.select("input_ids").rdd.map(lambda x: x[0]).collect(),
            "attention_mask": df.select("attention_mask").rdd.map(lambda x: x[0]).collect(),
            "token_type_ids": df.select("token_type_ids").rdd.map(lambda x: x[0]).collect()
        },
        df.select("label").rdd.map(lambda x: x[0]).collect()
    )).batch(batch_size)

    # 构建模型
    model = build_bert_model()

    # 广播初始参数到所有进程
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(f"bert_{task_name}_best.h5", save_best_only=True))

    # 训练模型
    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 if hvd.rank() == 0 else 0
    )

    # 仅主进程保存历史记录
    if hvd.rank() == 0:
        import pickle
        with open(f"train_history_{task_name}.pkl", "wb") as f:
            pickle.dump(history.history, f)
    return history


# 使用Spark分布式执行训练
if __name__ == "__main__":
    data_path = "/root/glue" if os.path.exists(
        "/root/glue") else "C:\\Users\\86198\\Desktop\\学习\\study\\课程学习\\计算机\\软件开发应用基础实践\\glue"

    # 配置分布式训练参数（4个Worker节点）
    distributor = TorchDistributor(
        num_processes=4,
        local_mode=False,
        use_gpu=True
    )

    # 训练SST-2任务
    distributor.run(train_task, "SST-2", data_path, epochs=3, batch_size=32)

    # 训练MRPC任务
    distributor.run(train_task, "MRPC", data_path, epochs=5, batch_size=16)
