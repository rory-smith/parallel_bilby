# All notable changes will be documented in this file

## v0.1.5 : 2020-09-08
- Remove redundant pool.wait [@conrad.chan]
- Add Max Iterations fix [@conrad.chan]
- Add `schwimmbad_fast.py` allowing MPI worker tasks workers to spawn at the start and persist until the end (no benefit to killing them early) [@conrad.chan]
- Fix trigger time parser to read strings [@colm.talbot]
- Remove mem per cpu default [@maite.mateu-lucena]
- Add nestcheck file option [@maite.mateu-lucena]
- Add constructing sky frame parameters [@colm.talbot]
- Change MPI isend --> send (to allow master task to finish sending messages to workers) [@conrad.chan]
- Disable the flag messages when timing is off [@conrad.chan]

## v0.1.4 : 2020-04-02
- Fix a bug in the reference frequency setting during post-processing. All runs prior to this can be fixed offline and should document the process here: https://wiki.ligo.org/CBC/ParamEst/Reference-frequencyBugInParallelBilbyMarch2020
- Adds an initial implementation of ptemcee
- Improvements to the dynesty checkpointing to reduce CPU idle time
- Parallelize initial points calculation
- Other bug fixes and improvements

## v0.1.3 : 2020-02-13
- Add documentation
- Add initialization of live points for constrained priors
- Add injection parameters to stored meta data
- Add a basic CI with pre commit

## v0.1.2 : 2020-01-30

- Updates bilby dependencies and improves sampler proposoal behavior

## v0.1.1 : 2020-01-21

Tighter integration with `bilby_pipe` and the addition of a full slurm-only scheduler. All inputs can now be specified via the ini file. For a complete set of instructions, see `parallel_bilby_generation --help`.

### Changes

- Improved integration of `parallel_bilby` with `bilby_pipe`. This effectively means that anything that can be done in `bilby_pipe` can also be done in `parallel_bilby`
- Adds a slurm submission script handler: see `{OUTDIR}/submit`
- Improvement to the bottleneck (if no new point is found, -inf is returned)
- Uses the bilby estimation of the autocorrelation time
- Adds version information tagging

## v0.1.0 : 2019-12-13
Initial first release with basic behaviour
