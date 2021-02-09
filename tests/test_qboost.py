# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import copy
import unittest

from sklearn.datasets import make_classification

from qboost import DecisionStumpClassifier, AllStumpsClassifier, QBoostClassifier


def _get_sample_data():
    X, y = make_classification(n_features=5)
    # Convert class labels to +/- 1
    y = y * 2 - 1
    return X, y


class DecisionStumpTest(unittest.TestCase):
    def test_decision_stump(self):

        X, y = _get_sample_data()

        stump = DecisionStumpClassifier(X, y, 2)

        # Basic sanity check:
        self.assertEqual(len(stump.predict(X)), len(X))


class EnsembleTest(unittest.TestCase):
    def test_ensemble(self):

        X, y = _get_sample_data()

        clf = AllStumpsClassifier(X, y)

        self.assertGreaterEqual(clf.score(X, y), 0)


class QBoostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = _get_sample_data()
        cls.clf = QBoostClassifier(cls.X, cls.y, 0.0)

    def test_qboost(self):
        self.assertGreaterEqual(self.clf.score(self.X, self.y), 0)

    def test_energy(self):
        """Sanity check that the BQM energy matches the squared error computed directly.

        Note this only works with lambda=0, and we also need to reset the offset to 0.
        """

        # BQM formulation is based on squared error with no offset, so
        # we need to reset the offset back to zero prior to computing
        # the squared error.

        # Create a shallow copy prior to modify the offset so as not
        # to interfere with other test cases using this same instance.
        clf = copy.copy(self.clf)
        clf.offset = 0.0
        
        squared_error = clf.squared_error(self.X, self.y)
        self.assertAlmostEqual(clf.energy * len(self.y), squared_error)
