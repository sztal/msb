Code for paper<br>
_Polarization and multiscale structural balance in signed networks_
======================================================================

This repository provides all instructions, code and data
necessary for reproducing the results from PAPER.

Code implementing main methods of the **Multiscale Semiwalk Balance** (MSB)
approach to quantifying degree of balance (DoB) in signed networks is organized
as a simple Python package `msb`, which can be installed directly from the
repository using PIP.

The `msb` package comes with basic documentation provided in docstrings
of classes and methods. Moreover, there is a simple tutorial notebook
(see _Notebooks_ section).

**NOTE.** The implementation provided here is quite flexible, efficient
and relatively well-documented, but it is not supposed to be a
production-grade software. It suffers from a few known issues
(which, however, should not affect the majority of the users), which
are described later in this README. A production-grade implementation
is planned and will be released as a part of a larger software package
in the future.


Repository structure
--------------------

    ├── data               <- directory with all data files used in the analyses
    ├── notebooks          <- directory with Jupyter notebooks implementing the analyses
    │   ├── performance    <- notebooks with accuracy and efficientcy analyses
    ├── msb                <- Python package implementing MSB approach
    ├── figs               <- Figures folder; created when needed
    ├── LICENSE
    ├── README.md
    ├── environment.yml    <- specification of Conda env for replicating the results
    └── pyproject.toml     <- metadata and build instructions for the 'msb' package


Requirements
------------

The code is known to run for Python3.10+ and should work for any OS.
Other dependencies are installed either when installing the `msb` package
or when setting up a Conda environment from the `environment.yml` file,
which also lists all dependencies.

Installing `msb` package from Github
------------------------------------

For using MSB approach in other applications/analyses, the most convenient
thing to do is to install `msb` package as a standalone Python package
directly from the Github repository.

```bash
pip install git+ssh://git@github.com/sztal/msb.git
```


Setting up the environment
--------------------------

Below are instructions for setting up an environment for replicating
the results. Here we assume using a Conda environment, but with some
additional tweaking any other approach (e.g. standard Python venv) should
also work.

```bash
# clone the repository from Github and enter it
git clone https://github.com/sztal/msb
cd msb

# Create Conda env
# By default it is named 'msb'
# but the name can be changed in 'environment.yml'
conda env create -f environment.yml
# Activate the environment
conda activate msb
# Install 'msb' package
pip install .
# Or install in 'editable' mode if needed
pip install --editable .
```

Notebooks
---------

All analyses are implemented in dedicated Jupyter notebooks.
Below is a detailed list.

* `1-tutorial`: this is a simple tutorial notebook showcasing main
  features of the `msb` package.
* `2-tests`: implements basic tests of the correctness of the computations
  implemented in the `msb` package. Main workhorse functions used for
  calculating DoB measures are validated against straightforward naive
  (i.e. not efficient) implementations.
* `3-contributions`: analysis of contribution scores.
* `4-monks`: re-analysis of the Sampson's Monastery data.
* `5-congress`: analysis of the polarization in the U.S. Congress.
* `performance`: this subdirectory stores notebooks evaluating
  accuracy and efficiency of the `msb` package. The notebooks depend
  on data, which is somewhat time-consuming to compute, so the raw data
  is computed with separate scripts and saved as pickle files.


Known issues
------------

The general design of the `msb` package is focused on high efficiency.
However, in order to avoid the need for temporarily allocating very large
arrays, some parts of the computations are not properly vectorized
and instead are carried out through pure Python loops. Thus, the package
is not yet fully optimized. However, in most cases it should be very
fast anyway (see `efficiency.ipynb` notebook in `notebooks/performance`).
Moreover, this is just a technical issue, which can be solved be
reimplementing the problematic parts, for instance as low-level C routines,
so it is not an inherent limitation of the MSB approach.

Furthermore, `pairwise_cohesion` method currently returns dense
$n \times n$ arrays, which may consume a lot of memory for large networks
(high $n$). In principle, it should be reimplemented as a linear operator
avoiding storing the whole dense array, but this is not how it is done
right now. As a result, `pairwise_cohesion` method applicability is
significantly limited by the available RAM memory and may not work
for large networks. However, as mentioned, this is also only a technical
issue that can be solved, so, again, it is not an inherent limitation of
the MSB approach.

Contact
-------

Szymon Talaga, `<stalaga@protonmail.com>`.
