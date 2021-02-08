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
from sklearn.tree import DecisionTreeClassifier


class DecisionStumpClassifier:
    """Decision tree classifier that operates on a single feature with a single splitting rule.
    
    The index of the feature used in the decision rule is stored
    relative to the original data frame.
    """
    def __init__(self, X, y, feature_index):
        """Initialize and fit the classifier.

        Args:
            X (array): 
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself uses only a single feature.
            y (array):
                1D array of class labels, as ints.  Labels should be
                +/- 1.
            feature_index (int):
                Index for the feature used by the weak classifier,
                relative to the overall data frame.
        """
        self.i = feature_index

        self.clf = DecisionTreeClassifier(max_depth=1)
        self.clf.fit(X[:, [feature_index]], y)

    def predict(self, X):
        """Predict class.
        
        Args:
            X (array):
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself will make a prediction based only a single
                feature.
        
        Returns:
            Array of class labels.
        """
        return self.clf.predict(X[:, [self.i]])
