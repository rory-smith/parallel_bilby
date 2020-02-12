=============
Installation
=============

Dependencies
------------

Install dependencies using

.. code-block:: console

   $ conda install -c conda-forge bilby_pipe schwimmbad

Installation
------------

Install the package locally with

.. code-block:: console

   $ python setup.py install


Executables
-----------

This gives you access to three executables

.. code-block:: console

   $ parallel_bilby_generation --help
   $ parallel_bilby_analysis --help
   $ parallel_bilby_retrieve_data --help

Roughly speaking, the generation executable is run locally to prepare the data
for analysis. It takes as input a :code:`gwf` file of the strain data, the PSD text
files and some configuration options such as which waveform model to use. The output
of the data generation step is a :code:`data_dump.pickle`.

This file is used as the input to the analysis step.
