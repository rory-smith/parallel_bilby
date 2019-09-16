# GW170817 IMRPhenomPv2_NRTidal with parallel bilby

## Prepare the data

First, download the 16384 Hz data available form GWOSC: https://www.gw-openscience.org/events/GW170817/, saving the files in a directory `raw_data`

```bash
$ cd raw_data
$ wget https://dcc.ligo.org/public/0146/P1700349/001/H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf
$ wget https://dcc.ligo.org/public/0146/P1700349/001/L-L1_LOSC_CLN_16_V1-1187007040-2048.gwf
$ wget https://dcc.ligo.org/public/0146/P1700349/001/V-V1_LOSC_CLN_16_V1-1187007040-2048.gwf
```

## Generation: preparing the data_dump

First, we generate the data dump file, 

```bash
$ parallel_bilby_generation GW170817.ini
```

## Analysis: run the analysis look under MPI

```bash
$ mkdir logs
$ sbatch slurm_submit_analysis.sh
```

## Post-process: run the post processing step

```bash
$ sbatch slurm_submit_postprocess.sh
```
