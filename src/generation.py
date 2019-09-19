#!/usr/bin/env python
"""
Generate/prepare data, likelihood, and priors for parallel runs
"""
import pickle

import numpy as np
import bilby
from bilby_pipe.parser import BilbyArgParser
from bilby_pipe.utils import convert_string_to_dict
from gwpy.timeseries import TimeSeries


logger = bilby.core.utils.logger


def get_args():
    parser = BilbyArgParser(
        usage=__doc__, ignore_unknown_config_file_keys=False, allow_abbrev=False
    )
    parser.add("ini", type=str, is_config_file=True, help="Configuration ini file")

    parser.add_argument(
        "-l", "--label", help="Label for the data", required=True)
    parser.add_argument(
        "-o", "--outdir", default="processed_data", help="Outdir for the processed data")

    parser.add_argument(
        "-t", "--trigger-time", required=True, help="Trigger time", type=float)
    parser.add_argument(
        "-d", "--duration", required=True, help="Segment duration", type=float)

    parser.add_argument(
        "--data-dict", type=convert_string_to_dict, required=True)
    parser.add_argument(
        "--channel-dict", type=convert_string_to_dict, required=True)
    parser.add_argument(
        "--psd-dict", type=convert_string_to_dict, required=True)

    parser.add_argument(
        "--prior-file", type=str, required=True)
    parser.add_argument(
        "--waveform-approximant", type=str, required=True)
    parser.add_argument(
        "--reference-frequency", default=20, help="The reference frequency")
    parser.add_argument(
        "--sampling-frequency", default=4096, help="The sampling frequency")
    parser.add_argument(
        "--minimum-frequency", default=20, help="The minimum frequency")
    parser.add_argument(
        "--maximum-frequency", default=2048, help="The maxmimum frequency")
    parser.add(
        "--calibration-model",
        type=str,
        default=None,
        choices=["CubicSpline"],
        help="Choice of calibration model, if None, no calibration is used",
    )
    parser.add(
        "--spline-calibration-envelope-dict",
        type=convert_string_to_dict,
        default=None,
        help=("Dictionary pointing to the spline calibration envelope files"),
    )
    parser.add(
        "--spline-calibration-nodes",
        type=int,
        default=5,
        help=("Number of calibration nodes"),
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    trigger_time = args.trigger_time
    duration = args.duration
    label = args.label
    outdir = args.outdir

    priors = bilby.gw.prior.PriorDict(filename=args.prior_file)
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - 0.1, trigger_time + 0.1, name="geocent_time")

    roll_off = 0.4  # Roll off duration of tukey window in seconds
    post_trigger_duration = 2  # Time between trigger time and end of segment
    end_time = trigger_time + post_trigger_duration
    start_time = end_time - duration

    ifo_list = bilby.gw.detector.InterferometerList([])
    for det in args.data_dict:
        logger.info(f"Reading in analysis data for ifo {det}")
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        channel = f"{det}:{args.channel_dict[det]}"
        data = TimeSeries.read(
            args.data_dict[det], channel=channel, start=start_time,
            end=end_time, dtype=np.float64, format="gwf.lalframe")
        data = data.resample(args.sampling_frequency)
        print(f"Data for {det} from {data.times[0]} to {data.times[-1]}")
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

    if "tidal" in args.waveform_approximant.lower():
        conv = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
    else:
        conv = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters

    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=conv,
        waveform_arguments={'waveform_approximant': args.waveform_approximant,
                            'reference_frequency': args.reference_frequency})

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list, waveform_generator, priors=priors, time_marginalization=False,
        phase_marginalization=True, distance_marginalization=True)

    data_dump_file = f"{outdir}/{label}_data_dump.pickle"
    data_dump = dict(likelihood=likelihood, priors=priors, args=args,
                     data_dump_file=data_dump_file)

    with open(data_dump_file, "wb+") as file:
        pickle.dump(data_dump, file)
