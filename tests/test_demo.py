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
import subprocess

# /path/to/demos/qboost/tests/test_demo.py
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IntegrationTest(unittest.TestCase):
    def _test_data_set(self, dataset):
        """Utility routine to run demo.py on given data set and check that it produces output."""

        demo_file = os.path.join(project_dir, 'demo.py')
        output = subprocess.check_output([sys.executable, demo_file, dataset])
        output = output.decode('utf-8') # Bytes to str

        self.assertIn("accu (train)", output)
        self.assertIn("accu (test)", output)
    
    def test_wisc(self):
        """Test that demo.py runs "wisc" data set without crashing and produces output."""
        self._test_data_set("wisc")
        
    def test_mnist(self):
        """Test that demo.py runs "mnist" data set without crashing and produces output."""
        self._test_data_set("mnist")
