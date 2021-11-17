import mpi4py

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

del mpi4py

from .analysis import main
