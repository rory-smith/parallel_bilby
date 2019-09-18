# GW170817 IMRPhenomPv2_NRTidal with parallel bilby

## Prepare the data

First, download the 16384 Hz data available form GWOSC:
https://www.gw-openscience.org/events/GW170817/, saving the files in a
directory `raw_data`. Here is a set of example commands to do so:

```bash
$ cd raw_data
$ wget https://dcc.ligo.org/public/0146/P1700349/001/H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf
$ wget https://dcc.ligo.org/public/0146/P1700349/001/L-L1_LOSC_CLN_16_V1-1187007040-2048.gwf
$ wget https://dcc.ligo.org/public/0146/P1700349/001/V-V1_LOSC_CLN_16_V1-1187007040-2048.gwf
```

## Generation: preparing the data_dump

The first step of the analysis is to prepare the data_dump file, this is a
python pickle file containing all the preprocessing before sampling. We use an
ini file to specify the configuration, in this case the file GW170817.ini:

```bash
$ parallel_bilby_generation GW170817.ini
```

As you'll see in the config file, we specify here which PSD, data, and which
waveform approximant to use. This step will build the lookup table for distance
marginalization if required and also generate some figures showing the data and
psd. It is recommended to sanity check these before continuing.

## Analysis: run the analysis under MPI

Once one has a data dump file, you can now run it under MPI. To check that
things work okay, on a head done you can run
```bash
$ mpirun parallel_bilby_analysis outdir/GW170817_data_dump.pickle
```
This will start sampling using the maximum number of cores available on a head
node, typically 3 or 4. To deploy the analysis at scale, you'll need to submit
it to the scheduler. We give an example below for a slurm scheduler.

On completion, this will generate a file `outdir/GW170817_result.json` which
contains the posterior as sampled.

## Post-process: run the post processing step

The sampling itself generates samples only in the sampled parameters. We
include a postprocessing step to generate samples in the parameter available
by conversion (i.e. componant masses) and the parameters which are marginalised
during sampling. This, like the analysis step, should be run under mpi, e.g.

```bash
$ mpirun parallel_bilby_postprocess outdir/GW170817_result.json
```

## Slurm submit script

For either the analysis or post-processing steps, you should submit the job to
run on a cluster, leveraging a number of parallel cores. As an example, here is
a slurm submit script for running the analysis step above using 8 tasks over
4 cores, giving a total of 32 cores abailable to MPI:

```ini
#!/bin/bash
#
#SBATCH --job-name=GW170817
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --output=logs/%A_%a.out

source </PATH/TO/YOUR/BASHRC>
conda activate <CONDA-ENVIRONME>
export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export MPI_PER_NODE=8
mpirun parallel_bilby_analysis outdir/GW170817_data_dump.pickle
```

Here, we have shown an example which first sources a conda environment, you
should edit this to ensure the correct environment with parallel_bilby and its
dependencies is available.

Having written this file, you can submit it with
```bash
$ mkdir logs/
$ sbatch slurm_submit_analysis.sh
```

## Configuration

See `--help`, on any of the three exectubles above to see options.
