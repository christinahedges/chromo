import numpy as np
import lightkurve as lk
import os
import warnings

from numpy import ma

from matplotlib import cbook
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from copy import deepcopy
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.io import fits

class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint


def _estimate_background(tpf):
    # Correct background
    flux = np.copy(tpf.flux)
    thumb = np.nanpercentile(flux, 95, axis=0)
    thumb[thumb > 1e3] = np.nan
    mask = np.nan_to_num(thumb) > np.nanpercentile(thumb, 5)
    mask |= ~np.isfinite(thumb)

    # Make sure to throw away nans
    mask[~np.isfinite(thumb)] = True

    # Build a background estimate using astropy sigma clipping
    mean, med, sigma = np.asarray([sigma_clipped_stats(f, mask=mask) for f in flux]).T

    return med

def background_correct(raw_tpf):
    bkg = _estimate_background(raw_tpf)
    hdu = deepcopy(raw_tpf.hdu)
    hdu[1].data['FLUX'][raw_tpf.quality_mask] -= np.atleast_3d(bkg).transpose([1, 2, 0])
    hdu[1].data['FLUX'][raw_tpf.quality_mask] -= np.min(hdu[1].data['FLUX'])
    fits.HDUList(hdus=list(hdu)).writeto('hack.fits', overwrite=True)
    tpf = lk.TessTargetPixelFile('hack.fits', quality_bitmask=raw_tpf.quality_bitmask)
    os.remove('hack.fits')
    return tpf

def poly_detrend(lc, eb_model, npoly=3, sigma=3):
    ''' Detrend a light curve with a simple third order polynomial
    '''
    clc = lc.copy()
    clc /= eb_model

    if np.any(np.diff(clc.time) > 0.5):
        split = np.where(np.diff(clc.time) > 0.5)[0][0]+1
        f = clc[:split].remove_outliers(sigma)
        corr = lk.LightCurve(clc[:split].time, np.polyval(np.polyfit(f.time, f.flux, npoly), clc[:split].time))

        f = clc[split:].remove_outliers(sigma)
        corr = corr.append(lk.LightCurve(clc[split:].time, np.polyval(np.polyfit(f.time, f.flux, npoly), clc[split:].time)))
    else:
        f = clc.remove_outliers(sigma)
        corr = lk.LightCurve(clc.time, np.polyval(np.polyfit(f.time, f.flux, npoly), clc.time))

    return corr



def movie(dat, phase, flux, title='', out='out.mp4', scale='linear', cbar_label='', **kwargs):
    '''Create an mp4 movie of a 3D array
    '''
    if scale == 'log':
        data = np.log10(np.copy(dat))
    else:
        data = dat

    s = data.shape[1:]
    aspect_ratio = s[1]/s[0]
    n = 6
    nr = int(np.round((n/aspect_ratio)))

    fig = plt.figure(figsize=(n*aspect_ratio,  n*1/aspect_ratio + 2))

    ax = plt.subplot2grid((nr, nr), (0, 0), rowspan=nr - 2, colspan=nr)
    ax.set_facecolor('lightgrey')
    ax_lc = plt.subplot2grid((nr, nr), (nr - 2, 0), colspan=nr, rowspan=2)


    if 'vmax' not in kwargs:
        kwargs['vmax'] = np.nanpercentile(data, 75)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = np.nanpercentile(data, 5)
    im1 = ax.imshow(data[0], origin='bottom', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    vmin = kwargs.pop('vmin', np.nanpercentile(data, 5))
    vmax = kwargs.pop('vmax', np.nanpercentile(data, 95))


#    ticks = np.append(np.round(np.linspace(vmin, 0, 4), 3)[:-1], np.round(np.linspace(0, vmax, 4), 3))
#    dt = (vmax - vmin)/8
#    ticks = np.append(np.round(np.arange(vmin, 0, dt), 3)[:-1], np.round(np.arange(0, vmax, dt), 3))
#    cbar = plt.colorbar(im1, cax=cax, ticks=ticks)
    cbar = plt.colorbar(im1, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(cbar_label)

    ax_lc.plot(phase, flux)
    ax_lc.set_ylabel('Normalized Flux')
    ax_lc.set_xlabel('Phase')

    point = ax_lc.scatter(phase[0], flux[0], s=10)
    def animate(i):
        im1.set_array(data[i])
        point.set_offsets((phase[i], flux[i]))

    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30)
    anim.save(out, dpi=150)

def plot_diagnostic(x_fold_b, resids, secondary_mask, folded_lightcurve, tpf,  aper=None, **kwargs):
    ''' Plot a nice crobat plot
    '''
    fig = plt.figure(figsize=(14, 9))

    ax = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2, fig=fig)
    folded_lightcurve.scatter(ax=ax, label='Period: {}d\nt0:      {}'.format(np.round(folded_lightcurve.meta['period'], 6),
                                                                        np.round(folded_lightcurve.meta['t0'], 6)))
    ax.set_xlabel('')
    ax.set_title(kwargs.pop('name', ''))

    ax = plt.subplot2grid((3, 4), (0, 3), fig=fig, rowspan=1)
    im = ax.imshow(np.log10(np.nanmedian(tpf.flux, axis=0)), cmap='viridis', vmin=0, vmax=4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('log$_{10]}$ Flux', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Flux')

    ax = plt.subplot2grid((3, 4), (1, 3), fig=fig, rowspan=1)
    labels = ['Mission', 'Sector', 'Camera', 'CCD', 'Row', 'Column', "RA", "Dec"]
    ax.axis('off')
    ax.table(cellText=np.atleast_2d(np.asarray([getattr(tpf, l.lower()) for l in labels])).T,
                    rowLabels=np.asarray(labels), colLabels=np.asarray(['Parameters']), colWidths=np.asarray([0.75]), loc='center')

    secondary_depth_resid = np.median(resids[secondary_mask], axis=0)

    vmin = np.min([0, np.nanmin(secondary_depth_resid[aper])])
    vmax = np.nanmax([0, np.nanmax(secondary_depth_resid[aper])])

    cmap = plt.get_cmap(kwargs.pop('cmap', 'RdBu'))
    norm = MidPointNorm(midpoint=0,
                        vmin=vmin,
                        vmax=vmax)
    cmap.set_bad('lightgrey', 1)
    dt = (vmax - vmin)/8
    ticks = np.append(np.round(np.arange(vmin, 0, dt), 3)[:-1], np.round(np.arange(0, vmax, dt), 3))
    ticks = np.unique(ticks)

    ax = plt.subplot2grid((3, 4), (2, 0), colspan=3, fig=fig)
#    ax.set_title('Residuals as a Function of Time', fontsize=10)
    for idx, l, n in zip(range(aper.sum()), deepcopy(resids[:, aper].T), secondary_depth_resid[aper]):
        if idx == 0:
            ax.plot(x_fold_b, l, color=cmap(norm(n)), zorder=n, label='Residuals')
        else:
            ax.plot(x_fold_b, l, color=cmap(norm(n)), zorder=n)
        #ax.plot(x_fold_b, (true_flux_b) * corr - corr + 1, color='k', zorder=n)
    ax.set_xlabel('Phase')
    ax.legend()
    #Horrible Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    #cbar.set_label('Measured Secondary Depth/\nTrue Secondary Depth')
    ax.set_ylabel('Pixel Light Curve Residuals')
#    ax.axvline(0, ls='--', color='k')

    ax = plt.subplot2grid((3, 4), (2, 3), colspan=1, fig=fig)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im = ax.imshow(secondary_depth_resid/(aper), cmap=cmap, norm=norm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=ticks)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Residual', fontsize=7)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Residuals During Secondary Eclipse', fontsize=10)

    return fig
