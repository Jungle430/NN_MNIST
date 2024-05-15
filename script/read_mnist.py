import os
import sys
import numpy as np
import idx2numpy
import threading
from typing import List


def save_images(image_file: str, images: np.ndarray, num: int) -> None:
    with open(image_file, "w") as f:
        for i in range(num):
            flattened_array: np.ndarray = images[i].reshape(-1)
            for k in range(flattened_array.shape[0]):
                if k != flattened_array.shape[0] - 1:
                    f.write(f"{flattened_array[k]},")
                else:
                    f.write(f"{flattened_array[k]}")
            f.write(f"\n")


def save_labels(labels_file: str, labels: np.ndarray, num: int) -> None:
    with open(labels_file, "w") as f:
        for i in range(num):
            f.write(f"{labels[i]}\n")


train_data_size_max: int = 60000
test_data_size_max: int = 10000

command_message: str = (
    "python3 read_mnist.py <train_data_set_size(0~60000)> <test_data_set_size>(0~10000)"
)

train_images_path: str = "../mnist_data/train-images-idx3-ubyte"
train_labels_path: str = "../mnist_data/train-labels-idx1-ubyte"
test_images_path: str = "../mnist_data/t10k-images-idx3-ubyte"
test_labels_path: str = "../mnist_data/t10k-labels-idx1-ubyte"

train_images_file: str = "../mnist_data/train_images.csv"
train_labels_file: str = "../mnist_data/train_labels.csv"
test_images_file: str = "../mnist_data/test_images.csv"
test_labels_file: str = "../mnist_data/test_labels.csv"

train_images: np.ndarray = idx2numpy.convert_from_file(train_images_path)
train_labels: np.ndarray = idx2numpy.convert_from_file(train_labels_path)
test_images: np.ndarray = idx2numpy.convert_from_file(test_images_path)
test_labels: np.ndarray = idx2numpy.convert_from_file(test_labels_path)

if __name__ == "__main__":
    os.system("rm -rf *.csv")

    if len(sys.argv) != 3:
        print(command_message)
        exit(-1)

    train_data_set_size: int = int(sys.argv[1])
    test_data_set_size: int = int(sys.argv[2])

    if not (
        0 <= train_data_set_size <= train_data_size_max
        and 0 <= test_data_set_size <= test_data_size_max
    ):
        print(command_message)
        exit(-1)

    works: List[threading.Thread] = []
    works.append(
        threading.Thread(
            target=save_images,
            args=(train_images_file, train_images, train_data_set_size),
        )
    )
    works.append(
        threading.Thread(
            target=save_labels,
            args=(train_labels_file, train_labels, train_data_set_size),
        )
    )
    works.append(
        threading.Thread(
            target=save_images, args=(test_images_file, test_images, test_data_set_size)
        )
    )
    works.append(
        threading.Thread(
            target=save_labels, args=(test_labels_file, test_labels, test_data_set_size)
        )
    )

    for work in works:
        work.start()

    for work in works:
        work.join()
