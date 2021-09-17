from setuptools import find_packages, setup

with open('README.rst') as f:
    readme = f.read()


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name='irmlearn',
    description='Infinite Relational Model in Python with scikit-learn like API.',
    long_description=readme,
    long_description_content_type="text/x-rst",
    license='MIT',
    author='Masafumi Abeta',
    author_email='ground0state@gmail.com',
    url='https://github.com/ground0state/irmlearn',
    packages=find_packages('src', exclude=['demo', 'tests']),
    package_dir={'': 'src'},
    install_requires=_requires_from_file('requirements.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
