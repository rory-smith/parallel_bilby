import dynesty.plotting as dyplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.sampler.dynesty import dynesty_stats_plot
from bilby.core.utils import logger

from ..utils import stopwatch

matplotlib.use("Agg")


@stopwatch
def plot_current_state(sampler, search_parameter_keys, outdir, label):
    labels = [label.replace("_", " ") for label in search_parameter_keys]
    try:
        filename = f"{outdir}/{label}_checkpoint_trace.png"
        fig = dyplot.traceplot(sampler.results, labels=labels)[0]
        fig.tight_layout()
        fig.savefig(filename)
    except (
        AssertionError,
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
    ) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty state plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_checkpoint_run.png"
        fig, axs = dyplot.runplot(sampler.results, mark_final_live=False)
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, np.linalg.linalg.LinAlgError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty run plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_checkpoint_stats.png"
        fig, _ = dynesty_stats_plot(sampler)
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty stats plot at checkpoint")
    finally:
        plt.close("all")
