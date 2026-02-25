import glob
from core.dataset import get_dataset
from training.trainer import train
from configs.config import CFG

files = glob.glob("data/tfrecords/*.tfrecord")

train_ds = get_dataset(files, CFG.batch_size)

train(train_ds)