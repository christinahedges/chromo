import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from copy import deepcopy
import logging
import warnings

from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.convolution import convolve, Box1DKernel, Box2DKernel, Gaussian2DKernel
from astropy.io import fits

from scipy.stats import pearsonr
from scipy.interpolate import interp1d

import lightkurve as lk

from .utils import *

log = logging.getLogger(__name__)
log.setLevel('INFO')


plt.style.use(lk.MPLSTYLE)

def analyze(raw_tpf, period, t0, name='target', aper=None, nb=100, build_movie=False, output_dir='', score_threshold=-30, saturation_aggression=5):
    '''Analyze a TESS EB, and diagnose the chromaticity.

    Parameters
    ----------

    raw_tpf: lightkurve.TessTargetPixelFile
        A TESS cut out TPF with an EB
    period : float
        Orbital period of EB
    t0 : float
        T0 Midpoint of the primary eclipse
    name : string
        Name of the target for plotting and file names
    aper : np.ndarray of bools, optional
        An optional aperture to use
    nb : int
        Number of bins to use when binning. Use a smaller value
        to bin more aggressively
    build_movie : bool
        Whether to build movies of the results. This will take longer.
    output_dir : string
        Where to output the resulting figures
    score_threshold: int
        TEMPORARY. A value to tune the aperture size. Set to a more negative number to get a larger aperture...
    saturation_aggression : int
        TEMPORARY. How tall a saturation column is. Set to higher values to remove more saturation.

    Returns
    -------

    results : dict
        A dictionary of results containing
            - data : the pixel light curves
            - errors : the errors on the pixel light curves
            - model : Model pixel light curves, based on the total light curve.
            - corr : the offsets identified in each pixel between the pixel lightcurve
                    and the total light curve.
            - aper : the aperture identified by `analyze`

    '''

    # Find saturated pixels
    saturated = np.nanpercentile(raw_tpf.flux, 90, axis=0) > 40000
    for idx, s in enumerate(saturated.T):
        saturated[:, idx] = (convolve(s, Box1DKernel(saturation_aggression)) > 1e-5)

    # Default aperture is all pixels
    if aper is None:
        aper = np.ones(raw_tpf.shape[1:], bool)


    # Background correct the raw TPF, see utils
    if isinstance(raw_tpf, lk.targetpixelfile.TessTargetPixelFile):
        tpf = background_correct(raw_tpf)
    else:
        tpf = raw_tpf

    # Create the light curve from ALL pixels in the aperture
    # This is the best estimate of the TESS light curve
    total_lightcurve = tpf.to_lightcurve(aperture_mask=aper).normalize()

    # Create the phase for the EB
    phase = (total_lightcurve.time - t0 - period/2) / period % 1 - 0.5

    # Measure the primary and secondary depth
    total_lightcurve_primary_depth = np.nanmedian(total_lightcurve.flux[np.abs(phase) < 0.02])
    total_lightcurve_secondary_depth = np.nanmedian(total_lightcurve.flux[np.abs(phase) > 0.48])

    # Bin total light curve in phase
    # --------
    # indexes to bin at
    inds = np.array_split(np.argsort(phase), np.linspace(0, len(phase), nb + 1, dtype=int))[1:-1]

    phase_binned = np.asarray([np.median(phase[ind]) for ind in inds])
    total_lightcurve_flux_binned = np.asarray([np.median(total_lightcurve.flux[ind]) for ind in inds])
    total_lightcurve_flux_binned /= np.nanmedian(total_lightcurve.flux)

    # Interpolate the binned total light curve to create a model of the EB
    func = interp1d(phase_binned[np.argsort(phase_binned)], total_lightcurve_flux_binned[np.argsort(phase_binned)],
                    kind='cubic', fill_value='extrapolate')
    eb_model = func(phase)

    # Remove long term trends, see utils
    total_lightcurve /= poly_detrend(total_lightcurve, eb_model).flux
    total_lightcurve_flux_binned = np.asarray([np.median(total_lightcurve.flux[ind]) for ind in inds])
    total_lightcurve_flux_binned /= np.nanmedian(total_lightcurve.flux)

    # Make deep copies, to ensure nothing gets editted in the lk.LightCurve
    time = deepcopy(tpf.time)
    flux = deepcopy(tpf.flux)
    flux_err = deepcopy(tpf.flux_err)
    total_lightcurve_flux = deepcopy(total_lightcurve.flux)

    # Calculate pixel light curves
    # ----------------------------


    # Arrays to store output
    data = np.zeros(flux.shape)
    errors = np.zeros(flux.shape)
    model = np.zeros(flux.shape)
    corr = np.zeros(flux.shape[1:])

    # For every pixel
    for jdx in tqdm(range(flux.shape[2]), desc='Calculating Pixel Light Curves'):
        for idx in range(flux.shape[1]):
            if (~np.isfinite(flux[:, idx, jdx])).all():
                continue

            # BUILD a lk object
            l1 = lk.LightCurve(time, flux[:, idx, jdx], flux_err=flux_err[:, idx, jdx])

            # Detrend long term
            l1 /= poly_detrend(l1, eb_model).flux
            l1 = l1.normalize()

            # Output
            data[:, idx, jdx] = l1.flux
            errors[:, idx, jdx] = l1.flux_err

            # Normalize to the primary depth!
            p = 1 - np.nanmedian(l1.flux[np.abs(phase) < 0.02])
            tp = 1 - np.nanmedian(total_lightcurve_flux[np.abs(phase) < 0.02])

            # Output normalization
            corr[idx, jdx] = (p/tp)
            corr_model_lc =  (total_lightcurve_flux) * corr[idx, jdx] - corr[idx, jdx] + 1

            # Out a model that has been normalized to each pixel primary depth.
            model[:, idx, jdx] = corr_model_lc


    # Calculated the correlation between the total light curve flux and each pixel
    correlation, score = np.zeros(data.shape[1:]), np.zeros(data.shape[1:])
    for idx in range(data.shape[1]):
        for jdx in range(data.shape[2]):
            correlation[idx, jdx], score[idx, jdx] = pearsonr(data[:, idx, jdx], total_lightcurve_flux)


    # Build an aperture, where the only pixels allowed are those where the correlation is strong.
    # If you want to use a larger aperture, change this score
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if np.any(aper & (np.log10(score) < score_threshold)):
            aper &= np.log10(score) < score_threshold

    # Phase and bin the data and model arrays
    data_binned = np.asarray([np.median(data[ind, :, :], axis=0) for ind in inds])
    model_binned = np.asarray([np.median(model[ind, :, :], axis=0) for ind in inds])

    # Residuals are the difference between the two
    resids = np.copy(data_binned) - np.atleast_3d(np.median(data, axis=0)).transpose([2, 0, 1])
    resids -= (np.copy(model_binned) -  np.atleast_3d(np.median(model, axis=0)).transpose([2, 0, 1]))


    # Make a crobat plot
    log.info('Plotting Crobat')
    folded_lightcurve = total_lightcurve.fold(period=period, t0=t0)
    folded_lightcurve.meta = {'period':period, 't0':t0}
    fig = plot_diagnostic(phase_binned, resids, np.abs(phase_binned) > 0.48, folded_lightcurve, tpf,
                            aper=aper & ~saturated, name=name)

    if tpf.mission.lower() == 'tess':
        fig.savefig('{}{}_{}.png'.format(output_dir, name.replace(' ', ''), 'sector{}'.format(tpf.sector)), dpi=200, bbox_inches='tight')
    if tpf.mission.lower() == 'kepler':
        fig.savefig('{}{}_{}.png'.format(output_dir, name.replace(' ', ''), 'quarter{}'.format(tpf.quarter)), dpi=200, bbox_inches='tight')
    if tpf.mission.lower() == 'k2':
        fig.savefig('{}{}_{}.png'.format(output_dir, name.replace(' ', ''), 'campaign{}'.format(tpf.campaign)), dpi=200, bbox_inches='tight')

    # If the user wants movies
    if build_movie:

        # Flux Movie
        # ----------
        ph, fl = phase_binned[np.argsort(phase_binned)], total_lightcurve_flux_binned[np.argsort(phase_binned)]

        log.info('\tBuilding Normalized Flux Animation')
        # Find the colour min/max for the colour bar
        fmin, fmax = np.nanpercentile(model_binned[:, aper & ~saturated], 1), np.nanmax([1, np.nanpercentile(model_binned[:, aper & ~saturated], 1)])
        movie(data_binned, ph, fl, cmap='viridis', vmin=fmin, vmax=fmax, out='{}{}_{}.mp4'.format(output_dir, name.replace(' ', ''), 'sector{}'.format(tpf.sector)),
                           title='Normalized Data', cbar_label='Normalized Flux')


        # Residuals Movie
        # ---------------
        cmap = plt.get_cmap('PuOr_r')
        vmin, vmax = np.min([0, np.nanpercentile(resids[:, aper & ~saturated], 1)]), np.nanmax([0, np.nanpercentile(resids[:, aper & ~saturated], 99)])
        vmax = np.max([np.abs(vmin), vmax])
        vmin = -vmax

        log.info('\tBuilding Residual Flux Animation')
        norm = MidPointNorm(midpoint=0, vmin=vmin, vmax=vmax)
        cmap.set_bad('lightgrey', 0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            movie(resids/np.atleast_3d(aper).transpose([2, 0, 1]), ph, fl,
                               cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, out='{}{}_{}_resids.mp4'.format(output_dir, name.replace(' ', ''), 'sector{}'.format(tpf.sector)),
                               title='Residuals', cbar_label='Resiudal')

    # Return the results as a dictionary
    results = {'data':data, 'errors':errors, 'model':model, 'corr':corr, "aper":aper & ~saturated}
    return results
