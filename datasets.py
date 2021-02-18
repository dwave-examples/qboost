#    Copyright 2018 D-Wave Systems Inc.

#    Licensed under the Apache License, Version 2.0 (the "License")
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http: // www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits


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
    if n_informative > n_features:
        raise ValueError("n_informative must be less than or equal to n_features")

    # Set up the centers so that only n_informative features have a
    # different mean for the two classes.
    class0_centers = np.zeros(n_features)
    class1_centers = np.zeros(n_features)
    class1_centers[:n_informative] = delta
    
    centers = np.vstack((class0_centers, class1_centers))
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)

    # Convert class labels to +/- 1
    y = y * 2 - 1

    return X, y


def get_handwritten_digits_data(class1, class2):
    """Retrieve handwritten digits data for two of the classes (digits).

    Args:
        class1 (int):
           First class label (digit) to include.  Between 0 and 9.
        class2 (int):
           Second class label (digit) to include.  Between 0 and 9.

    Returns:
        X (array of shape (n_samples, n_features))
            Feature vectors.
        y (array of shape (n_samples,):
            Class labels with values of +/- 1.
    """
    
    Xall, yall = load_digits(return_X_y=True)

    def extract_two_classes(c1, c2):
        idx = (yall == c1) | (yall == c2)

        X = Xall[idx,:]
        y = yall[idx]

        vals = np.unique(y)
        y[y == vals[0]] = -1
        y[y == vals[1]] = 1

        return X, y

    X, y = extract_two_classes(class1, class2)

    return X, y
