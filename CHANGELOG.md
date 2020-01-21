# All notable changes will be documented in this file

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
