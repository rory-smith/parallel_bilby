#!/usr/bin/env python
"""
Generate/prepare data, likelihood, and priors for parallel runs
"""
import argparse
import pickle
import os

import numpy as np
import bilby
from bilby.gw import conversion
from bilby_pipe.parser import BilbyArgParser
from bilby_pipe.utils import convert_string_to_dict
from gwpy.timeseries import TimeSeries


logger = bilby.core.utils.logger


class StoreBoolean(argparse.Action):
    """ argparse class for robust handling of booleans with configargparse

    When using configargparse, if the argument is setup with
    action="store_true", but the default is set to True, then there is no way,
    in the config file to switch the parameter off. To resolve this, this class
    handles the boolean properly.

    """

    def __call__(self, parser, namespace, value, option_string=None):
        value = str(value).lower()
        if value in ["true"]:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, False)


def get_args():
    parser = BilbyArgParser(
        usage=__doc__, ignore_unknown_config_file_keys=False,
        allow_abbrev=False
    )
    parser.add(
        "ini", type=str, is_config_file=True, help="Configuration ini file")
    parser.add_argument(
        "-l", "--label", help="Label for the data", required=True, type=str)
    parser.add_argument(
        "-o", "--outdir", default="processed_data",
        help="Outdir for the processed data", type=str)
    parser.add_argument(
        "-t", "--trigger-time", required=True, help="Trigger time", type=float)
    parser.add_argument(
        "-d", "--duration", required=True, help="Segment duration", type=float)
    parser.add(
        "--deltaT",
        type=float,
        default=0.2,
        help=(
            "The symmetric width (in s) around the trigger time to"
            " search over the coalesence time"
        ),
    )
    parser.add_argument(
        "--data-dict", type=convert_string_to_dict, required=True,
        help="Dictionary of paths to the data to analyse, e.g. {H1:data.gwf}")
    parser.add_argument(
        "--channel-dict", type=convert_string_to_dict, required=False,
        help=("Dictionary of channel names for each data file, used when data-"
              "dict points to gwf files"))
    parser.add_argument(
        "--psd-dict", type=convert_string_to_dict, required=True,
        help="Dictionary of paths to the relevant PSD files for each data file"
    )
    parser.add_argument(
        "--prior-file", type=str, required=True,
        help="Path to the Bilby prior file")
    parser.add_argument(
        "--waveform-approximant", type=str, required=True,
        help="Name of the waveform approximant")
    parser.add_argument(
        "--reference-frequency", default=20, help="The reference frequency",
        type=float)
    parser.add_argument(
<<<<<<< HEAD
        "--sampling-frequency", default=4096, help="The sampling frequency", type=float)
=======
        "--sampling-frequency", default=4096, help="The sampling frequency",
        type=float)
>>>>>>> 6adf42366225ed343eebd1730f30db2d32f28581
    parser.add_argument(
        "--minimum-frequency", default=20, help="The minimum frequency",
        type=float)
    parser.add_argument(
        "--maximum-frequency", default=2048, help="The maxmimum frequency",
        type=float)
    parser.add(
        "--calibration-model",
        default=None,
        choices=["CubicSpline"],
        help="Choice of calibration model, if None, no calibration is used",
        type=str,
    )
    parser.add(
        "--spline-calibration-envelope-dict",
        type=convert_string_to_dict,
        default=None,
        help=("Dictionary of paths to the spline calibration envelope files"),
    )
    parser.add(
        "--spline-calibration-nodes",
        type=int,
        default=10,
        help=("Number of calibration nodes"),
    )
    parser.add(
        "--distance-marginalization",
        action=StoreBoolean,
        required=True,
        help="Bool. If true, use a distance-marginalized likelihood",
    )
    parser.add(
        "--distance-marginalization-lookup-table",
        default=None,
        type=str,
        help="Path to the distance-marginalization lookup table",
    )
    parser.add(
        "--phase-marginalization",
        action=StoreBoolean,
        required=True,
        help="Bool. If true, use a phase-marginalized likelihood",
    )
    parser.add(
        "--time-marginalization",
        action=StoreBoolean,
        required=True,
        help="Bool. If true, use a time-marginalized likelihood",
    )
    parser.add(
        "--binary-neutron-star",
        action=StoreBoolean,
        default=False,
        help="If true, use a BNS source model function (i.e. with tides)",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    trigger_time = args.trigger_time
    duration = args.duration
    label = args.label
    outdir = args.outdir

    if args.binary_neutron_star or "tidal" in args.waveform_approximant.lower():
        conv = conversion.convert_to_lal_binary_neutron_star_parameters
        fdsm = bilby.gw.source.lal_binary_neutron_star
        priors = bilby.gw.prior.BNSPriorDict(args.prior_file)
    else:
        conv = conversion.convert_to_lal_binary_black_hole_parameters
        fdsm = bilby.gw.source.lal_binary_black_hole
        priors = bilby.gw.prior.BBHPriorDict(args.prior_file)

    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - args.deltaT / 2,
        trigger_time + args.deltaT / 2,
        name="geocent_time")

    roll_off = 0.4  # Roll off duration of tukey window in seconds
    post_trigger_duration = 2  # Time between trigger time and end of segment
    end_time = trigger_time + post_trigger_duration
    start_time = end_time - duration

    ifo_list = bilby.gw.detector.InterferometerList([])
    for det in args.data_dict:
        logger.info(f"Reading in analysis data for ifo {det}")
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        if "gwf" in os.path.splitext(args.data_dict[det])[1]:
            channel = f"{det}:{args.channel_dict[det]}"
            data = TimeSeries.read(
                args.data_dict[det], channel=channel, start=start_time,
                end=end_time, dtype=np.float64, format="gwf.lalframe")
        elif "hdf5" in os.path.splitext(args.data_dict[det])[1]:
            data = TimeSeries.read(
                args.data_dict[det], start=start_time,
                end=end_time, format="hdf5")
        elif "txt" in os.path.splitext(args.data_dict[det])[1]:
            data = TimeSeries.read(args.data_dict[det])
            data = data[data.times.value >= start_time]
            data = data[data.times.value < end_time]

        else:
            raise ValueError(f"Input file for detector {det} not understood")

        data = data.resample(args.sampling_frequency)
        logger.info(f"Data for {det} from {data.times[0]} to {data.times[-1]}")
        ifo.strain_data.minimum_frequency = args.minimum_frequency
        ifo.strain_data.maximum_frequency = args.maximum_frequency
        ifo.strain_data.roll_off = roll_off
        ifo.strain_data.set_from_gwpy_timeseries(data)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            psd_file=args.psd_dict[det])
        ifo_list.append(ifo)

        if args.calibration_model == "CubicSpline":
            ifo.calibration_model = bilby.gw.calibration.CubicSpline(
                prefix="recalib_{}_".format(ifo.name),
                minimum_frequency=ifo.minimum_frequency,
                maximum_frequency=ifo.maximum_frequency,
                n_points=args.spline_calibration_nodes,
            )

            priors.update(
                bilby.gw.prior.CalibrationPriorDict.from_envelope_file(
                    args.spline_calibration_envelope_dict[det],
                    minimum_frequency=ifo.minimum_frequency,
                    maximum_frequency=ifo.maximum_frequency,
                    n_nodes=args.spline_calibration_nodes,
                    label=det,
                )
            )

    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
    ifo_list.plot_data(outdir=outdir, label=label)

    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=fdsm,
        parameter_conversion=conv,
        start_time=start_time,
        waveform_arguments={'waveform_approximant': args.waveform_approximant,
                            'reference_frequency': args.reference_frequency})

    logger.info(
        "Setting up likelihood with marginalizations: "
        f"distance={args.distance_marginalization} "
        f"time={args.time_marginalization} "
        f"phase={args.phase_marginalization} ")

    # This is done before instantiating the likelihood so that it is the full prior
    prior_file = f"{outdir}/{label}_prior.json"
    priors.to_json(outdir=outdir, label=label)

    # We build the likelihood here to ensure the distance marginalization exist
    # before sampling
    bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list, waveform_generator, priors=priors,
        time_marginalization=args.time_marginalization,
        phase_marginalization=args.phase_marginalization,
        distance_marginalization=args.distance_marginalization,
        distance_marginalization_lookup_table=args.distance_marginalization_lookup_table)

    data_dump_file = f"{outdir}/{label}_data_dump.pickle"
    data_dump = dict(
        waveform_generator=waveform_generator, ifo_list=ifo_list,
        prior_file=prior_file, args=args, data_dump_file=data_dump_file)

    with open(data_dump_file, "wb+") as file:
        pickle.dump(data_dump, file)

    logger.info("Generation done: now run:\nmpirun parallel_bilby_analysis {}"
                .format(data_dump_file))
