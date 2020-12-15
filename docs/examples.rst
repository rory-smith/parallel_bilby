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

            $ bash outdir/submit/bash_<label>.sh

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

The folder has three ini examples:

#. GW150914 ini

#. GW170817 ini

#. Instructions to setup multiple injection inis

.. _Parallel Bilby Examples Folder: https://git.ligo.org/uploads/-/system/project/avatar/1846/bilby.jpg?width=40