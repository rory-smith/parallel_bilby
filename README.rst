parallel_bilby
==============

A python package to simplify the process of running parallel_bilby 

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

Roughly speaking, the generation executable is run locally to prepare the data
for analysis. It takes as input a :code:`gwf` file and configuration options
such as which waveform model to use. The output, a :code:`data_dump.pickle` file
is then the input to the analysis.
