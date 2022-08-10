=============
Installation
=============

Stable Installation
-------------------
Install the most recent stable version of :code:`Parallel Bilby` with :code:`pip`:

.. code-block:: console

    $ pip install parallel_bilby

Alternatively, you can install :code:`Parallel Bilby` with :code:`conda`

.. code-block:: console

    $ conda install -c conda-forge parallel-bilby


Dependencies
------------

Install dependencies using

.. code-block:: console

   $ conda install -c conda-forge bilby_pipe schwimmbad

Development Install
-------------------

Install the package locally with

.. code-block:: console

   $ python setup.py develop

This allows you to install :code:`Parallel Bilby` in a way that allows you edit your
code after installation to environment, and have the changes take effect immediately
(without re-installation).

Executables
-----------

Installing :code:`Parallel Bilby` gives you access to two executables

.. code-block:: console

   $ parallel_bilby_generation --help
   $ parallel_bilby_analysis --help

Roughly speaking, the generation executable is run locally to prepare the data
for analysis. It takes as input a :code:`gwf` file of the strain data, the PSD text
files and some configuration options such as which waveform model to use. The output
of the data generation step is a :code:`data_dump.pickle`.

This file is used as the input to the analysis step.
