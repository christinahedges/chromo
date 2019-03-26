import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.convolution import convolve, Box1DKernel, Box2DKernel, Gaussian2DKernel

import lightkurve as lk

from .utils import poly_detrend


def analyze(tpf, period, t0, aper=None):
    '''
    '''
    x_fold = (tpf.time - t0 - period/2) / period % 1 - 0.5

    # Find saturated pixels
    saturated = np.nanmax(flux, axis=0) > 100000
    for idx, s in enumerate(saturated.T):
        saturated[:, idx] = (convolve(s, Box1DKernel(5)) > 1e-5)

    saturated |= np.nanmax(flux, axis=0) > 50000
    for idx, s in enumerate(saturated.T):
        saturated[:, idx] = (convolve(s, Box1DKernel(10)) > 1e-5)

    if aper is None:
        aper = np.ones(flux.shape[1:], bool)

    true = tpf.to_lightcurve(aperture_mask=aper).normalize()



    # Correct background
    flux = np.copy(tpf.flux)
    thumb = np.nanpercentile(flux, 95, axis=0)
    mask = thumb > np.nanpercentile(thumb, 30)

    # Make sure to throw away nans
    mask[~np.isfinite(thumb)] = True

    # Build a background estimate using astropy sigma clipping
    mean, med, sigma = np.asarray([sigma_clipped_stats(f, mask=mask) for f in flux]).T

    # Remove the median background
    flux = tpf.flux - np.atleast_3d(med).transpose([1, 2, 0])

    # HAVE to add the minimum back in, otherwise some flux values are negative!
    flux -= np.min(flux)

    flux_err = tpf.flux_err

    return flux
