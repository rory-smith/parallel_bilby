==================
Usage and Examples
==================


Usage notes
-----------

The steps to analyse data with :code:`Parallel Bilby` are:

#. Ini Creation:

    Create an :code:`ini` with paths to the `prior`, `PSD` and `data` files, along with other `kwargs`.

#. Parallel Bilby Generation:

    Setup your :code:`Parallel Bilby` jobs with

        .. code-block:: console

            $ parallel_bilby_generation <ini file>

    This generates

    * Plots of the `PSD` (review before submitting your job)

    * :code:`Slurm` submission scripts

    * a :code:`data dump` pickle (object packed with the `PSD`, `data`, etc)

#. Parallel Bilby Analysis:

    To submit the :code:`Slurm` jobs on a cluster, run

    .. code-block:: console

        $ bash outdir/submit/bash_<label>.sh

    Alternatively, to run locally without submitting a job, check the :code:`bash` file
    for the required command. It should look something like:

    .. code-block:: console

       $ mpirun parallel_bilby_analysis outdir/data/<label>_data_dump.pickle --label <label> --outdir outdir/result --sampling-seed 1234`



Example ini files
-----------------

Refer to the `Parallel Bilby Examples Folder`_ for example :code:`ini` files along with :code:`Jupyter Notebooks`
explaining how to set up :code:`Parallel Bilby` jobs.

The folder has three examples:


`GW150914`_
~~~~~~~~~~~
To analyse GW150914 with Parallel Bilby you may use the following :code:`ini` file.
An explanation of the :code:`ini` file's contents are presented in the `GW150914 tutorial.ipynb`_.

.. include:: ../examples/GW150914_IMRPhenomPv2/GW150914.ini
    :code: INI

In this example we automate the analysis data download process.
We also include the priors for the analysis inside the :code:`ini` file.


`GW170817`_
~~~~~~~~~~~
To analyse GW170817 with Parallel Bilby you may use the following :code:`ini` file.

.. include:: ../examples/GW170817_IMRPhenomPv2_NRTidal/GW170817.ini
    :code: INI

In this example we require the user to manually download the data for analysis.
The priors are contained in a separate file for this analysis.

Again, an explanation of the :code:`ini` file's contents are presented in the `GW170817 tutorial.ipynb`_,
along with commands needed to download the analysis data.

.. include:: ../examples/GW170817_IMRPhenomPv2_NRTidal/GW170817.ini
    :code: jupyter no


`Multiple Injections`_
~~~~~~~~~~~~~~~~~~~~~~
You may need to analyse multiple injections with Parallel Bilby.
The `Multiple Injections`_ folder contains some code to help create submission files for each injection.


.. _Parallel Bilby Examples Folder: https://git.ligo.org/lscsoft/parallel_bilby/-/tree/master/examples
.. _GW150914: https://git.ligo.org/lscsoft/parallel_bilby/-/tree/master/examples/GW150914_IMRPhenomPv2/
.. _GW170817: https://git.ligo.org/lscsoft/parallel_bilby/-/tree/master/examples/GW170817_IMRPhenomPv2_NRTidal/
.. _Multiple Injections: https://git.ligo.org/lscsoft/parallel_bilby/-/tree/master/examples/multiple_pbilby_injections/
.. _GW150914 tutorial.ipynb: https://git.ligo.org/lscsoft/parallel_bilby/-/tree/master/examples/GW150914_IMRPhenomPv2/tutorial.ipynb
.. _GW170817 tutorial.ipynb: https://git.ligo.org/lscsoft/parallel_bilby/-/tree/master/examples/GW170817_IMRPhenomPv2_NRTidal/tutorial.ipynb
