# -*- coding: utf-8 -*-
"""
stardust.utils
==============
Shared helper utilities.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _make_interp2d(x_pts, y_pts, Z, method='linear'):
    """
    Drop-in replacement for the removed scipy.interpolate.interp2d.

    Returns a callable  f(xi, yi)  that replicates the old interp2d API:
        f(xi, yi) -> ndarray of shape (len(yi), len(xi))

    Parameters
    ----------
    x_pts  : 1-D array  – x-axis points (e.g. grain sizes  [µm])
    y_pts  : 1-D array  – y-axis points (e.g. radii or wavelengths)
    Z      : 2-D array  – values at grid, shape (len(y_pts), len(x_pts))
    method : str        – interpolation method ('linear', 'nearest', ...)
    """
    rgi = RegularGridInterpolator(
        (np.asarray(y_pts, dtype=float), np.asarray(x_pts, dtype=float)),
        np.asarray(Z, dtype=float),
        method=method, bounds_error=False, fill_value=None,
    )

    def _call(xi, yi):
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        yi = np.atleast_1d(np.asarray(yi, dtype=float))
        yg, xg = np.meshgrid(yi, xi, indexing='ij')
        pts = np.column_stack([yg.ravel(), xg.ravel()])
        return rgi(pts).reshape(len(yi), len(xi))

    return _call
