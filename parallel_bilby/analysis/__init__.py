import mpi4py

from .analysis import analysis_runner, main

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

del mpi4py
