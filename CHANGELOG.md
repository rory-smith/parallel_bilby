# All notable changes will be documented in this file

## v1.0.0 : 2020-11-25
- Update GW examples + add tutorial notebooks (e59e7068)
- Add multiple injection example (!83)
- Fix badges (!82)
- Fix missing keys from marginalised runs (!79)

## v0.1.6 : 2020-09-08
- Rename `src` dir to `parallel_bilby` dir (!76)
- Add complete config ini to outdir (!75)
- Add ROQ likelihood (!60)

## v0.1.5 : 2020-09-08
- Change MPI isend --> send (to allow master task to finish sending messages to workers) (!71)
- Disable the flag messages when timing is off (!71)
- Pre-commit issues (!69)
- Add constructing sky frame parameters (!68)
- Fix trigger time parser to read strings (!67)
- Add `schwimmbad_fast.py` allowing MPI worker tasks workers to spawn at the start and persist until the end (no benefit to killing them early) (!65)
- Remove mem per cpu default (!64)
- Add nestcheck file option (!64)
- Add Max Iterations fix (!62)
- Remove redundant pool.wait (!61)
- Increase time between checkpoints (!59)
- Add timing logs between checkpointing (!58)

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
