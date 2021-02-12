import numpy as np

from sklearn.datasets import make_blobs


def make_blob_data(n_samples=100, n_features=5, n_informative=2, delta=1):
    """Generate sample data based on isotropic Gaussians with a specified number of class-informative features.

    Args:
        n_samples (int):
            Number of samples.
        n_features (int):
            Number of features.
        n_informative (int):
            Number of informative features.
        delta (float):
            Difference in mean values of the informative features
            between the two classes.  (Note that all features have a
            standard deviation of 1).

    Returns:
        X (array of shape (n_samples, n_features))
            Feature vectors.
        y (array of shape (n_samples,):
            Class labels with values of +/- 1.
    """

    # Set up the centers so that only n_informative features have a
    # different mean for the two classes.
    class0_centers = np.zeros(n_features)
    class1_centers = np.zeros(n_features)
    class1_centers[:2] = delta
    
    centers = np.vstack((class0_centers, class1_centers))
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)

    # Convert class labels to +/- 1
    y = y * 2 - 1

    return X, y

