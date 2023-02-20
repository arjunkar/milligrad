import numpy as np

class DataConstructor:
    def __init__(self, image_size, train_image_path, train_label_path,
                test_image_path, test_label_path) -> None:
        self.train_images = self.read_image(train_image_path, image_size)
        self.train_labels = self.read_label(train_label_path)
        self.test_images = self.read_image(test_image_path, image_size)
        self.test_labels = self.read_label(test_label_path)

    def read_image(self, path, image_size):
        f = open(path, 'rb')
        f.read(16) # skip header
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, image_size, image_size)
        f.close()
        return data

    def read_label(self, path):
        f = open(path,'rb')
        f.read(8) # skip header
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        f.close()
        return data

    def train(self):
        return self.train_images, self.train_labels

    def test(self):
        return self.test_images, self.test_labels

class DataLoader():
    def __init__(self, data, targets, batch_size) -> None:
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.index = 0
        self.norm = 255

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        size = min(self.batch_size, len(self.data) - self.index)
        X = self.data[self.index:self.index+size] / self.norm
        y = self.targets[self.index:self.index+size]
        self.index += size
        return (X,y)