import numba
import numpy as np
from numba import float32, float64, guvectorize, vectorize
import xarray as xr
    
    
@guvectorize(
    [(float32[:], float64[:], float64[:]), (float64[:], float64[:], float64[:]),],
    "(i), (i) -> ()",
    nopython=True,
)
def _gufunc_hmxl(b, z, out):
    out[:] = np.nan
    
    mask = ~np.isnan(b)
    b = b[mask]
    z = z[mask]
    
    if len(b) == 0:
        return

    max_dbdz_surf = ((b - b[0]) / z).max()

    dbdz = np.diff(b) / np.diff(z)

    for idx in range(len(dbdz)):
        if dbdz[idx] > max_dbdz_surf:
            break

    ip = idx + 1
    if dbdz[idx] - dbdz[ip] < 0:
        ip = idx - 1
    
    if np.abs(dbdz[idx] - dbdz[ip]) < 1e-10:
        hmxl = z[ip]
    else:
        hmxl = z[ip] + np.abs(
            (z[idx] - z[ip]) / (dbdz[idx] - dbdz[ip]) * (max_dbdz_surf - dbdz[ip])
        )

    out[:] = hmxl


def calc_hmxl(pdens):
    z = pdens.cf["vertical"]
    zdim = z.name
    return xr.apply_ufunc(
        _gufunc_hmxl,
        -9.81/1025 * pdens,
        -np.abs(z),
        input_core_dims=[(zdim,), (zdim,)],
        dask="parallelized",
        output_dtypes=[z.dtype],
    )


def pdens(S, theta):
    """ Wright 97 EOS from https://mom6-analysiscookbook.readthedocs.io/en/latest/05_Buoyancy_Geostrophic_shear.html """

    @vectorize(["float32(float32, float32)"])
    def eos(S, theta):
        # --- Define constants (Table 1 Column 4, Wright 1997, J. Ocean Tech.)---
        a0 = 7.057924e-4
        a1 = 3.480336e-7
        a2 = -1.112733e-7

        b0 = 5.790749e8
        b1 = 3.516535e6
        b2 = -4.002714e4
        b3 = 2.084372e2
        b4 = 5.944068e5
        b5 = -9.643486e3

        c0 = 1.704853e5
        c1 = 7.904722e2
        c2 = -7.984422
        c3 = 5.140652e-2
        c4 = -2.302158e2
        c5 = -3.079464

        # To compute potential density keep pressure p = 100 kpa
        # S in standard salinity units psu, theta in DegC, p in pascals

        p = 100000.0
        alpha0 = a0 + a1 * theta + a2 * S
        p0 = (
            b0
            + b1 * theta
            + b2 * theta ** 2
            + b3 * theta ** 3
            + b4 * S
            + b5 * theta * S
        )
        lambd = (
            c0
            + c1 * theta
            + c2 * theta ** 2
            + c3 * theta ** 3
            + c4 * S
            + c5 * theta * S
        )

        pot_dens = (p + p0) / (lambd + alpha0 * (p + p0))

        return pot_dens

    return xr.apply_ufunc(eos, S, theta, dask="parallelized", output_dtypes=[S.dtype])


def calc_mld(pden):

    drho = pden - pden.cf.isel(Z=0)
    return xr.where(drho > 0.015, np.abs(drho.cf["Z"]), np.nan).cf.min("Z")