[![Linux/Mac/Windows build status](https://circleci.com/gh/dwave-examples/qboost.svg?style=svg)](https://circleci.com/gh/dwave-examples/qboost)

# Qboost

The D-Wave quantum computer has been widely studied as a discrete optimization
engine that accepts any problem formulated as quadratic unconstrained binary
optimization (QUBO). In 2008, Google and D-Wave published a paper,
[Training a Binary Classifier with the Quantum Adiabatic Algorithm](https://arxiv.org/pdf/0811.0416.pdf), which describes how the Qboost
ensemble method makes binary classification amenable to quantum computing: the problem is formulated as a thresholded linear superposition of a set of
weak classifiers and the D-Wave quantum computer is used to optimize the
weights in a learning process that strives to minimize the training error
and number of weak classifiers

This code demonstrates the use of the D-Wave system to solve a binary
classification problem using the Qboost algorithm.

## Disclaimer

This demo and its code are intended for demonstrative purposes only and are not
designed for performance.

## Usage

A minimal working example using the main interface function can be seen by
running:

```
python demo.py  --wisc --mnist
```

## References

H. Neven, V. S. Denchev, G. Rose, and W. G. Macready, "Training a Binary
Classifier with the Quantum Adiabatic Algorithm", [arXiv:0811.0416v1](https://arxiv.org/pdf/0811.0416.pdf)

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
