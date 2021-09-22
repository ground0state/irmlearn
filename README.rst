irmlearn
============

|image0| |image1| 

irmlearn is a algorithms for Infinite Relational Model.

This contains these techniques.

- IRM
- Poisson IRM

Dependencies
------------------------

The required dependencies to use irmlearn are,

- scikit-learn
- numpy
- scipy

You also need matplotlib, seaborn to run the demo and pytest to run the tests.

install
------------

.. code:: bash

    pip install irmlearn


USAGE
------------

We have posted a usage example in the github's demo folder.

IRM
~~~~~~~

.. code:: python

    from irmlearn import IRM


    alpha = 1.5
    a = 0.1
    b = 0.1
    max_iter = 300

    model = IRM(alpha, a, b, max_iter, verbose=True, use_best_iter=True)

    model.fit(X)


Poisson IRM
~~~~~~~~~~~~~~

.. code:: python

    from irmlearn import PoissonIRM


    alpha = 0.5
    a = 5
    b = 5
    max_iter = 300

    model = PoissonIRM(alpha, a, b, max_iter, verbose=True, use_best_iter=True)

    model.fit(X)



License
------------

This code is licensed under MIT License.

Test
------------

.. code:: python

    python setup.py test

.. |image0| image:: https://img.shields.io/badge/dynamic/json.svg?label=version&colorB=5f9ea0&query=$.version&uri=https://raw.githubusercontent.com/ground0state/irmlearn/main/package.json&style=plastic
.. |image1| image:: https://static.pepy.tech/personalized-badge/irmlearn?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
 :target: https://pepy.tech/project/irmlearn