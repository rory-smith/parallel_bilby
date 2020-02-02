import gwpy

ini = "tests/test_files/test.ini"
strain_file = "tests/test_files/strain_data/strain.hdf5"
psd_strain_file = "tests/test_files/strain_data/strain_for_psd.hdf5"


def get_timeseries():
    return (
        gwpy.timeseries.TimeSeries.read(strain_file),
        gwpy.timeseries.TimeSeries.read(psd_strain_file),
    )
