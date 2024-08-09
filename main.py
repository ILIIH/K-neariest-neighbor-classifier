from matplotlib import pyplot as plt
import numpy as np
from dataset import load_svhn
from knn_classifier import knn_classifier

def main():
    train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)
    visualise_data(train_X,train_y)
    prepare_data(train_X,train_y,test_y,test_X)


def prepare_data(train_X ,train_y,test_y,test_X):
    binary_train_mask = (train_y == 0) | (train_y == 9)
    binary_train_X = train_X[binary_train_mask]
    binary_train_y = train_y[binary_train_mask] == 0

    binary_test_mask = (test_y == 0) | (test_y == 9)
    binary_test_X = test_X[binary_test_mask]
    binary_test_y = test_y[binary_test_mask] == 0

    # Reshape to 1-dimensional array [num_samples, 32*32*3]
    binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1).astype(int)
    binary_train_Y = binary_train_y.reshape(binary_train_y.shape[0], -1).astype(int)

    binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1).astype(int)
    binary_test_Y = binary_test_y.reshape(binary_test_y.shape[0], -1).astype(int)

    classifier = knn_classifier()
    distancies = classifier.compute_distances_two_loops(binary_train_X,binary_train_Y,binary_test_X,binary_test_Y)

    print(distancies[0][0])

    
def visualise_data(train_X, train_Y):
    samples_per_class = 5  
    plot_index = 1
    for example_index in range(samples_per_class):
        for class_index in range(10):
            plt.subplot(5, 10, plot_index)
            image = train_X[train_Y == class_index][example_index]
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
            plot_index += 1

if __name__ == "__main__":
    main()
