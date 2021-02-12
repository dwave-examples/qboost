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
from sklearn.model_selection import train_test_split

from qboost import qboost_lambda_sweep
from datasets import make_blob_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run QBoost example")
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(title='dataset', description='dataset to use', dest='dataset', required=True)
    sp_blobs = subparsers.add_parser('blobs', help='blobs data set')
    
    sp_blobs.add_argument('--num-samples', type=int, default=2000, help='number of samples (default: %(default)s)')
    sp_blobs.add_argument('--num-features', type=int, default=10, help='number of features (default: %(default)s)')
    sp_blobs.add_argument('--num-informative', type=int, default=2, help='number of informative features (default: %(default)s)')

    args = parser.parse_args()


    if args.dataset == 'blobs':
        n_samples = args.num_samples
        n_features = args.num_features
        n_informative = args.num_informative

        X, y = make_blob_data(n_samples=n_samples, n_features=n_features, n_informative=n_informative)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


        normalized_lambdas = np.linspace(0.0, 0.5, 10)
        lambdas = normalized_lambdas / n_features
        qboost, lam = qboost_lambda_sweep(X_train, y_train, lambdas, verbose=args.verbose)

        print('best lambda:', lam)
        print('informative features:', list(range(n_informative)))
        print('   selected features:', qboost.get_selected_features())

        print('score on test set:', qboost.score(X_test, y_test))
