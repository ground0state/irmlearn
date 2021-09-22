# irmlearn

![](https://img.shields.io/badge/dynamic/json.svg?label=version&colorB=5f9ea0&query=$.version&uri=https://raw.githubusercontent.com/ground0state/irmlearn/main/package.json&style=plastic)
[![Downloads](https://pepy.tech/badge/irmlearn)](https://pepy.tech/project/irmlearn)



irmlearn is a algorithms for Infinite Relational Model.

This contains these techniques.

- IRM
- Poisson IRM

## Dependencies

The required dependencies to use irmlearn are,

- scikit-learn
- numpy
- scipy

You also need matplotlib, seaborn to run the demo and pytest to run the tests.

## install

```bash
pip install irmlearn
```

## USAGE

We have posted a usage example in the demo folder.

### IRM

```python
from irmlearn import IRM


alpha = 1.5
a = 0.1
b = 0.1
max_iter = 300

model = IRM(alpha, a, b, max_iter, verbose=True, use_best_iter=True)

model.fit(X)
```

### Poisson IRM

```python
from irmlearn import PoissonIRM


alpha = 0.5
a = 5
b = 5
max_iter = 300

model = PoissonIRM(alpha, a, b, max_iter, verbose=True, use_best_iter=True)

model.fit(X)
```

## License

This code is licensed under MIT License.

## Test

```python
python setup.py test
```
