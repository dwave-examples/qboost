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


class EnsembleClassifier:
    """Ensemble of weak classifiers."""
    def __init__(self, weak_classifiers, weights, weak_classifier_scaling, offset=1e-9):
        """Initialize ensemble from list of weak classifiers and weights.
        
        Args:
            weak_classifiers (list):
                List of classifier instances.
            weights (array):
                Weights associated with the weak classifiers.
            weak_classifier_scaling (float):
                Scaling for weak classifier outputs.
            offset (float):
                Offset value for ensemble classifier.  The default
                value is a small positive number used to prevent
                ambiguous 0 predictions when weak classifiers exactly
                balance each other out.
        """
        self.classifiers = weak_classifiers
        self.w = weights
        self.weak_clf_scale = weak_classifier_scaling
        self.offset = offset

    def _build_H(self, X):
        """Construct matrix of weak classifier predictions on given set of input vectors."""
        H = np.array([clf.predict(X) for clf in self.classifiers], dtype=float).T

        # Rescale H
        H *= self.weak_clf_scale

        return H

    def predict(self, X):
        """Compute ensemble prediction.

        Note that this function returns the numerical value of the
        ensemble predictor, not the class label.  The predicted class
        is sign(predict()).
        """
        H = self._build_H(X)

        # If we've already filtered out those with w=0 and we are only
        # using binary weights, this is just a sum
        preds = np.dot(H, self.w)
        return preds - self.offset

    def score(self, X, y):
        if sum(self.w) == 0:
            # Avoid difficulties that occur with handling this below
            return 0.0

        preds = self.predict(X)
        # If > is used instead of >=, then weak classifiers whose
        # votes exactly balance out end up not counting as a
        # prediction for either class.  This can occur when using
        # binary weights with an even number of classifiers.

        # Make sure we don't have any 0 preds.  Currently trying to
        # avoid these by using a small offset.  Otherwise, predictions
        # that equal 0, which occur when classifiers balance each
        # other out, can be problematic for scoring.
        assert all(preds != 0)

        return np.mean( np.sign(preds)*y > 0 )

    def squared_error(self, X, y):
        p = self.predict(X)
        return sum( (p - y)**2 )

    def fit_offset(self, X):
        """Fit offset value based on class-balanced feature vectors.

        Currently, this assumes that the feature vectors in X
        correspond to an even split between both classes.
        """
        self.offset = 0.0
        # Todo: review whether it would be appropriate to subtract
        # mean(y) here to account for unbalanced classes.
        self.offset = np.mean(self.predict(X))


class AllStumpsClassifier(EnsembleClassifier):
    """Ensemble classifier with one decision stump for each feature."""
    def __init__(self, X, y):

        num_featuers = np.size(X, 1)
        classifiers = [DecisionStumpClassifier(X, y, i) for i in range(num_featuers)]
        # The scaling is arbitrary in this case and does not affect the predictions.

        super().__init__(classifiers, np.ones(num_featuers), 1/num_featuers)
        self.fit_offset(X)
