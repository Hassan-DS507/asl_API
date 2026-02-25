import tensorflow as tf

def get_dataset(files, batch_size):

    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds