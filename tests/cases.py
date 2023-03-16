import numpy as np
from bilby.core.prior import Constraint, Cosine, PowerLaw, Sine, Uniform
from tests.utils import _Run


class FastRun(_Run):
    test_label = "fast"
    generation_args = dict(
        nlive=10,
        dlogz=100.0,
        nact=1,
        walks=1,
        sampling_seed=0,
        generation_seed=0,
        trigger_time="0",
        zero_noise=True,
        detectors=["H1"],
        injection_dict={
            "chirp_mass": 28.0,
            "mass_ratio": 1.0,
            "a_1": 0,
            "a_2": 0,
            "tilt_1": 0.0,
            "tilt_2": 0.0,
            "phi_12": 0.0,
            "phi_jl": 0.0,
            "luminosity_distance": 1000,
            "dec": 0,
            "ra": 0,
            "theta_jn": 0,
            "psi": 0,
            "phase": 0,
            "geocent_time": 0.0,
        },
        prior_dict=(
            "{chirp_mass = Uniform(name='chirp_mass', minimum=25, maximum=35), "
            "mass-ratio = 1, a_1 = 0, a_2 = 0, tilt_1 = 0, tilt_2 = 0, "
            "phi_12 = 0, phi_jl = 0, luminosity_distance = 1000, "
            "dec =  0, ra =  0, theta_jn = 0, psi = 0, phase = 0, geocent_time=0,}"
        ),
        no_plot=True,
    )


_roq_path = "tests/test_files/roq"


class ROQRun(_Run):
    test_label = "ROQ"
    generation_args = {
        "nlive": 20,
        "nact": 50,
        "sampling_seed": 0,
        "generation_seed": 0,
        "min_eff": 3.0,
        "calibration_model": "CubicSpline",
        "spline_calibration_envelope_dict": f"{{H1:{_roq_path}/GWTC1_GW150914_H_CalEnv.txt,"
        + f"L1:{_roq_path}/GWTC1_GW150914_L_CalEnv.txt}}",
        "spline_calibration_nodes": 10,
        "trigger_time": "1126259462.3910",
        "data_dict": f"{{H1:{_roq_path}/dataH1.gwf, L1:{_roq_path}/dataL1.gwf}}",
        "channel_dict": "{H1:DCS-CALIB_STRAIN_C02, L1:DCS-CALIB_STRAIN_C02}",
        "detectors": ["H1", "L1"],
        "psd_dict": f"{{H1:{_roq_path}/h1_psd.dat, L1:{_roq_path}/l1_psd.dat}}",
        "sampling_frequency": 2048.0,
        "maximum_frequency": "1024",
        "likelihood_type": "ROQGravitationalWaveTransient",
        "roq_folder": f"{_roq_path}/4s/",
        "prior_file": "4s",
    }


class GW150914Run(_Run):
    test_label = "GW150914"
    generation_args = {
        "trigger_time": "1126259462.4",
        "gaussian_noise": True,
        "detectors": ["H1", "L1"],
        "psd_dict": "{H1=examples/GW150914/psd_data/h1_psd.txt,"
        + "L1=examples/GW150914/psd_data/l1_psd.txt}",
        "phase_marginalization": True,
        "time_marginalization": True,
        "sampling_seed": 0,
        "generation_seed": 0,
        "prior_dict": dict(
            mass_ratio=Uniform(name="mass_ratio", minimum=0.125, maximum=1),
            chirp_mass=Uniform(name="chirp_mass", minimum=25, maximum=31),
            mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
            mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
            a_1=Uniform(name="a_1", minimum=0, maximum=0.99),
            a_2=Uniform(name="a_2", minimum=0, maximum=0.99),
            tilt_1=Sine(name="tilt_1"),
            tilt_2=Sine(name="tilt_2"),
            phi_12=Uniform(
                name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            phi_jl=Uniform(
                name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            luminosity_distance=PowerLaw(
                alpha=2, name="luminosity_distance", minimum=50, maximum=2000
            ),
            dec=Cosine(name="dec"),
            ra=Uniform(name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"),
            theta_jn=Sine(name="theta_jn"),
            psi=Uniform(name="psi", minimum=0, maximum=np.pi, boundary="periodic"),
            phase=Uniform(
                name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
        ),
    }
