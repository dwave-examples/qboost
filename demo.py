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
from sklearn.datasets import load_digits
try:
    import matplotlib.pyplot as plt
except ImportError:
    # Not required for demo
    pass

from qboost import qboost_lambda_sweep
from datasets import make_blob_data, get_handwritten_digits_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run QBoost example", epilog="Information about additional options that are specific to the data set can be obtained using either 'demo.py blobs -h' or 'demo.py digits -h'.")
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(title='dataset', description='dataset to use', dest='dataset', required=True)

    sp_blobs = subparsers.add_parser('blobs', help='blobs data set')
    sp_blobs.add_argument('--num-samples', type=int, default=2000, help='number of samples (default: %(default)s)')
    sp_blobs.add_argument('--num-features', type=int, default=10, help='number of features (default: %(default)s)')
    sp_blobs.add_argument('--num-informative', type=int, default=2, help='number of informative features (default: %(default)s)')

    sp_digits = subparsers.add_parser('digits', help='handwritten digits data set')
    sp_digits.add_argument('--digit1', type=int, default=0, choices=range(10), help='first digit to include (default: %(default)s)')
    sp_digits.add_argument('--digit2', type=int, default=1, choices=range(10), help='second digit to include (default: %(default)s)')
    sp_digits.add_argument('--plot-digits', action='store_true', help='plot a random sample of each digit')

    args = parser.parse_args()


    if args.dataset == 'blobs':
        n_samples = args.num_samples
        n_features = args.num_features
        n_informative = args.num_informative

        X, y = make_blob_data(n_samples=n_samples, n_features=n_features, n_informative=n_informative)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        # See Boyda et al. (2017), Eq. (17) regarding normalization
        normalized_lambdas = np.linspace(0.0, 0.5, 10)
        lambdas = normalized_lambdas / n_features
        qboost, lam = qboost_lambda_sweep(X_train, y_train, lambdas, verbose=args.verbose)

        print('best lambda:', lam)
        print('informative features:', list(range(n_informative)))
        print('   selected features:', qboost.get_selected_features())

        print('score on test set:', qboost.score(X_test, y_test))


    if args.dataset == 'digits':
        if args.digit1 == args.digit2:
            raise ValueError("must use two different digits")

        X, y = get_handwritten_digits_data(args.digit1, args.digit2)
        n_features = np.size(X, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        print('Number of features:', np.size(X, 1))
        print('Number of training samples:', len(X_train))
        print('Number of test samples:', len(X_test))

        # See Boyda et al. (2017), Eq. (17) regarding normalization
        normalized_lambdas = np.linspace(0.0, 1.75, 10)
        lambdas = normalized_lambdas / n_features
        qboost, lam = qboost_lambda_sweep(X_train, y_train, lambdas, verbose=args.verbose)

        print('best lambda:', lam)
        print('number of selected features:', len(qboost.get_selected_features()))

        print('score on test set:', qboost.score(X_test, y_test))

        if args.plot_digits:
            digits = load_digits()

            images1 = [image for image,target in zip(digits.images, digits.target) if target == args.digit1]
            images2 = [image for image,target in zip(digits.images, digits.target) if target == args.digit2]
            
            f, axes = plt.subplots(1, 2)

            # Select a random image from each set to show:
            i1 = np.random.choice(len(images1))
            i2 = np.random.choice(len(images2))
            for ax, image in zip(axes, (images1[i1], images2[i2])):
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

            plt.show()

