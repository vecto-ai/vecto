from vecto.data import Dataset

ds = Dataset("/home/blackbird/Projects/NLP/datasets/STSA/binary")

print(ds)
print(ds.metadata)
train = ds.get_train()
print(train)
