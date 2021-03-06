[![Linux/Mac/Windows build status](
  https://circleci.com/gh/dwave-examples/qboost.svg?style=svg)](
  https://circleci.com/gh/dwave-examples/qboost)

# QBoost

In machine learning, boosting methods are used to combine a set of simple,
"weak" predictors in such a way as to produce a more powerful, "strong"
predictor.  The QBoost method selects weak classifiers by minimizing the squared
difference between observed and predicted labels.  The resulting optimization
problem can be formulated as a binary quadratic model and solved on the D-Wave
system.  This approach has been explored previously in the literature, for
example Refs. [1-4].

This example illustrates the application of a QBoost formulation on two
illustrative data sets.

## Usage

The demonstration includes two sample data sets, which can be run with the
following commands:

```bash
python demo.py blobs
```

Or:

```bash
python demo.py digits
```

### Blobs data set

This is a simple synthetically generated data set for illustrative purposes,
based on the scikit-learn function `sklearn.datasets.make_blobs`.  Data for
classification are randomly generated with two classes and an arbitrary number
of samples, total features, and informative features.  The features are randomly
generated from Gaussian distributions.  For each informative feature, the mean
value of the feature differs between the two target classes, while the
non-informative features are sampled from the same probability distribution in
each class.

Each run of the demo will randomly generate the data set and split it into
training and test sets. The results of the demo will print the indices of the
informative features, the features that are selected by QBoost, and the accuracy
of the resulting ensemble classifier on the test set.

The values of the number of samples, number of features, and number of
informative features can be controlled using command line arguments, as
described in the help:

```bash
python demo.py blobs -h
```

### Digits data set

The digits data is based on the well-known [handwritten digits data
set](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset).
Each instance is a digitized image of a handwritten digit.  The images consist
of 8x8 grayscale pixel values, which are represented as 64 features.  The goal
is to construct a classifier that can predict the digit that is represented by
an image.

This demonstration constructs a binary classification problem using images for
any two of the available ten digits, as specified by the `--digit1` and
`--digit2` command line options (the defaults are 0 and 1).  As with the blobs
data set, the available data are split into training and test sets.  The
demonstration will print the number of features selected by QBoost, along with
the prediction accuracy on the test set.

The `--plot-digits` option is also provided to display one instance of the
images for each of the two selected digits.  The displayed images are chosen
randomly from the available data.

The following command displays the available options for use with the digits
data set:

```bash
python demo.py digits -h
```

## Code Overview

Boosting methods construct a strong classifier by intelligently combining a set
of weak classifiers.  Given a set of `N` weak classifiers, QBoost solves an
optimization problem to select weak classifiers.  The objective of this
optimization problem is to minimize the strong classifier's squared loss between
actual and predicted targets.  A regularization term is also included to
penalize complex models that include more weak classifiers.  The resulting
optimization problem can be written as [1]:

![Objective](images/objective.png)

where `s` is an index over training instances, `i` is an index over features,
`h_i(x_s)` denote the weak classifiers, `y_s` denote the observed targets, `w_i`
are the weights to be determined, and lambda is the regularization parameter,
which multiplies the L0 norm of the weight vector.  In this demonstration, the
weights are treated as binary variables that take a value of either 0 or 1.  The
weak classifiers are constructed from a series of single-feature decision tree
rules, also known as decision stumps.  Following Refs. [1,4], the output of each
weak classifier is scaled to `-1/N` and `+1/N`.

By default, the demonstration is run with a fixed value of the regularization
parameter, lambda.  If the `--cross-validation` option is used, then a simple
parameter sweep is performed, and the value of lambda is selected on the basis
of providing the highest accuracy on a validation set.  This requires repeatedly
solving the optimization problem, and the demonstration may take several minutes
to run when this option is used.

## Disclaimer

This demo and its code are intended for demonstration purposes only and are not
designed for performance.  There are a variety of additional considerations, not
addressed in the example code, which should be considered as part of a more
detailed implementation.  These include:

- This example uses simple decision stumps as weak classifiers, but in principle
  the method can be applied to any collection of weak classifiers.
- Iteration to select weak classifiers in batches [2].
- More rigorous determination of the regularization parameter.
- Further evaluation of weak classifier output scaling.  Most prior work in the
  literature scales the weak classifier outputs by the number of weak
  classifiers.  Ref. [4] suggests that scaling by the square root instead
  is more natural.
- This example employs a bit depth of 1, so that the weight associated with each
  classifier is limited to 0 or 1.  It is possible to increase the bit depth,
  for example as discussed Ref. [1].  Nevertheless, it has been reported
  that low bit depths can improve performance, particularly on smaller training
  sets.

## References

[1] Neven, H., Denchev, V. S., Rose, G., and Macready, W. G.  Training a Binary
Classifier with the Quantum Adiabatic Algorithm, 2008,
[arXiv:0811.0416v1](https://arxiv.org/pdf/0811.0416.pdf)

[2] Neven, H., Denchev, V. S., Rose, G., and Macready, W. G.  QBoost: Large Scale
Classifier Training with Adiabatic Quantum Optimization, Journal of Machine
Learning Research: Workshop and Conference Proceedings, 2012.  URL:
http://proceedings.mlr.press/v25/neven12/neven12.pdf.

[3] Mott, A., Job, J., Vlimant, J.-R., Lidar, D., and Spiropulu, M.  Solving a Higgs
optimization problem with quantum annealing for machine learning.  Nature,
Vol. 550, 2017, doi:10.1038/nature24047.

[4] Boyda, E., Basu, S., Ganguly, S., Michaelis, A., Mukhopadhyay, S., and Nemani,
R. R.  Deploying a quantum annealing processor to detect tree cover in aerial
imagery of California.  PLoS One, 2017, doi:10.1371/journal.pone.0172505.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
