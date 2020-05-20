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
import subprocess
import unittest
import sys
# /path/to/demos/qboost/tests/test_demo.py
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDemo(unittest.TestCase):
    def test_smoke(self):
        """run demo.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'demo.py')
        subprocess.check_output([sys.executable, demo_file, "--mnist", "--wisc"])

    def test_qboost(self):
        """run demo.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'demo.py')
        output = subprocess.check_output([sys.executable, demo_file, "--wisc", "--mnist"])
        output = str(output).upper()
        if os.getenv('DEBUG_OUTPUT'):
            print("Example output \n" + output)

        with self.subTest(msg="Verify if output contains 'accu (train):' \n"):
            self.assertIn("accu (train):".upper(), output)
        with self.subTest(msg="Verify if output contains 'DecisionTree' \n"):
            self.assertIn("DecisionTree".upper(), output)
        with self.subTest(msg="Verify if error string contains in output \n"):
            self.assertNotIn("ERROR", output)
        with self.subTest(msg="Verify if warning string contains in output \n"):
            self.assertNotIn("WARNING", output)
