import tensorflow as tf

ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "pytorch/data/*.txt",
    batch_size=10,
    column_names=["Open","High","Low","Close","tal","kal","sal","tas","kas","sas"],
    header=False, 
    label_name="tal",
    num_epochs=1,
    num_parallel_reads=1,
    shuffle=False,
    shuffle_buffer_size=10000)
for features in ds.take(1):
    print(features)