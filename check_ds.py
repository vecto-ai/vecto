from vecto.data import Dataset
path = "/mnt/storage/data/NLP/datasets/text_classification/SST-2"
#path = "/home/blackbird/Projects/NLP/datasets/STSA/binary"

ds = Dataset(path)

print(ds)
print(ds.metadata)
train = ds.get_train()
print(train)
