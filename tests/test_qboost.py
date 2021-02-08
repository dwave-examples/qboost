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
import unittest

from sklearn.datasets import make_classification

from qboost import DecisionStumpClassifier


class DecisionStumpTest(unittest.TestCase):
    def test_decision_stump(self):

        X, y = make_classification(n_features=5)

        stump = DecisionStumpClassifier(X, y, 2)

        # Basic sanity check:
        self.assertEqual(len(stump.predict(X)), len(X))


