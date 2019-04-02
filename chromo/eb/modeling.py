'''Model EBs using starry'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

from scipy.optimize import minimize

import lightkurve as lk
from . import eb
from . import utils
from .. import PACKAGEDIR

import starry

plt.style.use(lk.MPLSTYLE)

log = logging.getLogger()
log.setLevel('INFO')

_TESS_BANDPASS = pd.read_csv('{}/data/tess-response-function-v1.0.csv'.format(PACKAGEDIR), comment='#', header=None)

labels = ['radius', 'L', 'ellipsoidal amplitude', 'reflection_amplitude',
          'primary_u_0', 'primary_u_1', 'secondary_u_0', 'secondary_u_1',
          'tref', 'period', 'inclination', 'omega', 'eccentricity', 'a']


def ellipsoidal_variation(A_1, A_2, phase, phase2):
    ''' Basic model for ellipsoidal variations

    Parameters
    ----------
    A_1 : float
        Amplitude 1
    A_2 : float
        Amplitude 2
    phase : np.ndarray
        Phase, folded such that the primary is at 0
    phase2 : np.ndarray
        Phase, folded such that the secondary is at 0

    Returns
    -------
    ellipsoidal_variation : np.ndarray
        Normalized ellipsoidal variation
    '''
    return A_1 * np.cos(4 * np.pi * phase) + A_2 * np.cos(2 * np.pi * phase2)

class ModelException(Exception):
    '''Raised if there is a problem with the model'''
    pass

class Model(object):
    def _get_bounds(self, params):
            guess_bounds = [lambda x: (0.3, 0.9), lambda x: (0.05, 0.5), lambda x: (-0.1, 0), lambda x: (-0.1, 0),
                      lambda x: (0, 1), lambda x: (0, 1), lambda x: (0, 1), lambda x: (0, 1),
                     lambda x: (x - 0.01, x + 0.01), lambda x: (x - 0.02, x + 0.02),
                     lambda x: (0, 90), lambda x: (0, 30), lambda x: (0, 0.02), lambda x: (x - 3, x + 3)]

            bound_value = [b(params[l]) for l, b in zip(labels, guess_bounds)]
            bounds = {}
            for label, bound in zip(labels, bound_value):
                bounds[label] = bound
            return bounds

    def _get_system_state(self):
        ''' Get the star system state '''
        state = [self.system.secondaries[0].r, self.system.secondaries[0].L, None, None,
                 self.system.primary.u[0], self.system.primary.u[1], self.system.secondaries[0].u[0], self.system.secondaries[0].u[1],
                 self.system.secondaries[0].tref, self.system.secondaries[0].porb, self.system.secondaries[0].inc, self.system.secondaries[0].w,
                 self.system.secondaries[0].ecc, self.system.secondaries[0].a]
        state_dict = {}
        for s, label in zip(state, labels):
            state_dict[label] = s
        return state_dict


    def _set_system_state(self, params):
        ''' Set the starry system state'''
        self.system.secondaries[0].r = params['radius']
        self.system.secondaries[0].L = params['L']
        self.system.primary[1] = params['primary_u_0']
        self.system.primary[2] = params['primary_u_1']
        self.system.secondaries[0][1] = params['secondary_u_0']
        self.system.secondaries[0][2] = params['secondary_u_0']
        self.system.secondaries[0].tref = params['tref']
        self.system.secondaries[0].porb = params['period']
        self.system.secondaries[0].inc = params['inclination']
        self.system.secondaries[0].w = params['omega']
        self.system.secondaries[0].ecc = params['eccentricity']
        self.system.secondaries[0].a = params['a']

    def __init__(self, init, nb=100):
        '''starry modeling class'''
        self.primary = starry.kepler.Primary(lmax=2)
        self.secondary = starry.kepler.Secondary(lmax=2)
        self.system = starry.kepler.System(self.primary, self.secondary)
        if not isinstance(init, dict):
            raise ModelException('Please pass in a dictionary')
        if not ((np.in1d(labels, list(init.keys())).all()) & (np.in1d(list(init.keys()), labels).all())):
            raise ModelException('All parameters from the following list must be specified in the dictionary.\n* '
                                    + '\n* '.join(['{}'.format(x) for x in labels]))
        self.initial_guess = init
        self.best_guess = None
        self.bounds = self._get_bounds(init)
        self.nb = nb

        self._set_system_state(self.initial_guess)

    def __repr__(self):
        df = pd.DataFrame(columns=['Inital Guess', 'Bounds', 'Best Guess'], index=np.asarray(labels))
        for idx, label, bound in zip(range(len(labels)), labels, self.bounds.values()):
            df.loc[label, 'Inital Guess'] = self.initial_guess[label]
            if isinstance(bound[0], float):
                df.loc[label, 'Bounds'] = '({:2.4f}, {:2.4f})'.format(bound[0], bound[1])
            else:
                df.loc[label, 'Bounds'] = '({}, {})'.format(bound[0], bound[1])
            if self.best_guess is not None:
                if label in self.best_guess:
                    df.loc[label, 'Best Guess'] = self.best_guess[label]
                else:
                    df.loc[label, 'Best Guess'] = np.nan
        return tabulate(df, headers=df.columns, tablefmt='psql')

    def _build_phase(self, lc, period, tref):
        x_fold = (lc.time - tref - period/2) / period % 1 - 0.5
        x_fold_2 = np.copy(x_fold) + 0.5
        x_fold_2[x_fold > 0] -= 1

        # THIS IS APPROXIMATE AND WRONG
        k = (np.abs(x_fold) > 0.1) & (np.abs(x_fold) < 0.4)
        return x_fold, x_fold_2, k

    def _clean_lc(self, lc, period, tref):
        lc = lc.copy()
        x_fold, x_fold_2, k = self._build_phase(lc, period, tref)
        inds = np.array_split(np.argsort(x_fold), np.linspace(0, len(x_fold), self.nb + 1, dtype=int))[1:-1]
        x_fold_b = np.asarray([np.median(x_fold[ind]) for ind in inds])
        flux_b = np.asarray([np.median(lc.flux[ind]) for ind in inds])
        flux_b /= np.nanmedian(lc.flux)

        func = interp1d(x_fold_b[np.argsort(x_fold_b)], flux_b[np.argsort(x_fold_b)],
                        kind='cubic', fill_value='extrapolate')
        eb_model = func(x_fold)
        lc /= utils.poly_detrend(lc, eb_model).flux
        return lc


    def _likelihood(self, params, lc, x_fold, x_fold_2, k, return_model=False):
        self.system.secondaries[0].r = params[0]
        self.system.secondaries[0].L = params[1]

        if (len(params) > 2):
            ev = ellipsoidal_variation(params[2], params[3], x_fold + 0.5, x_fold_2 + 0.5)
            self.system.secondaries[0].tref = params[8]
            self.system.secondaries[0].porb = params[9]
            self.system.secondaries[0].inc = params[10]
            self.system.secondaries[0].w = params[11]
            self.system.secondaries[0].ecc = params[12]
            self.system.secondaries[0].a = params[13]

        else:
            ev = ellipsoidal_variation(result.x[2], result.x[3], x_fold + 0.5, x_fold_2 + 0.5)

        self.system.compute(lc.time)
        model = self.system.lightcurve/np.median(self.system.lightcurve[k]) + ev
        if return_model:
            return model

        chi = (1/len(lc.flux)) * np.sum((lc.flux - model)**2/(lc.flux_err**2))
        return chi


    def fit(self, lc, fix_orbital=False):
        period = self.initial_guess['period']
        tref = self.initial_guess['tref']

        lc = self._clean_lc(lc, period, tref)

        x_fold, x_fold_2, k = self._build_phase(lc, period, tref)

        params = list(self.initial_guess.values())
        if fix_orbital:
            params = params[:2]

        log.info('Fitting')
#        print(self.best_guess)
        result = minimize(self._likelihood, list(self.initial_guess.values()), method='TNC', bounds=list(self.bounds.values()), args=(lc, x_fold, x_fold_2, k))
        self._minimize_result = result
        bg = {}
        for idx, l in enumerate(labels):
            if fix_orbital:
                if (idx <= 1):
                    bg[l] = result.x[idx]
                else:
                    bg[l] = np.copy(self.best_guess[l])
            else:
                bg[l] = result.x[idx]

        self.best_guess = bg

    def plot(self, lc, ax=None):
        ''' Plot eb model'''
        if ax is None:
            _, ax = plt.subplots(1)
        if self.best_guess is not None:
            if np.isfinite(self.best_guess['period']):
                period, tref = self.best_guess['period'], self.best_guess['tref']
            else:
                period, tref = self.initial_guess['period'], self.initial_guess['tref']
        else:
            period, tref = self.initial_guess['period'], self.initial_guess['tref']

        lc = self._clean_lc(lc, period, tref)
        x_fold, x_fold_2, k = self._build_phase(lc, period, tref)

        lc.fold(period, tref).scatter(ax=ax, label='Data')

        state = self._get_system_state()
        mlc = lk.LightCurve(lc.time, self._likelihood(list(self.initial_guess.values()), lc, x_fold, x_fold_2, k, return_model=True))
        mlc.fold(period, tref).scatter(ax=ax, label='Initial Guess')

        if self.best_guess is not None:
            if np.isfinite(np.asarray(list(self.best_guess.values()))).all():
                mlc = lk.LightCurve(lc.time, self._likelihood(list(self.best_guess.values()), lc, x_fold, x_fold_2, k, return_model=True))
                mlc.fold(period, tref).scatter(ax=ax, label='Best Guess')

        self._set_system_state(state)

        return ax

    @property
    def radius(self):
        if (best_guess is None):
            raise ModelException('No best guess is available. Please run the `fit` method.')
        return best_guess['radius']

    @property
    def L(self):
        if (best_guess is None):
            raise ModelException('No best guess is available. Please run the `fit` method.')
        return best_guess['L']
