from mpi4py import MPI


def mpi_master(func):
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            f = func(*args, **kwargs)
        else:
            f = None
        comm.Barrier()
        return f

    return wrapper
