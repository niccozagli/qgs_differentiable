"""Output of the symbolic tendencies: Ocean-Atmosphere (MAOOAM) model.

Generates symbolic (parameter-dependent) tendencies for a fixed qgs MAOOAM
configuration and writes them to a standalone Numba tendency function file.

This is the export pipeline only: it runs up to and including writing the
generated tendency function (`funcs`) to disk.
"""

import sys
import os

sys.path.extend([os.path.abspath('../../')])

import numpy as np

# Importing the model's modules
from qgs.params.params import QgParams
from qgs.functions.symbolic_tendencies import create_symbolic_tendencies
from qgs.inner_products.analytic import (
    AtmosphericAnalyticInnerProducts,
    OceanicAnalyticInnerProducts,
)

def main():
    # -----------------------------------------------------------------------
    # Systems definition
    # -----------------------------------------------------------------------

    # Time parameters
    dt = 0.1

    # Model parameters instantiation with some non-default specs
    model_parameters = QgParams()

    # Mode truncation at the wavenumber 2 in both x and y spatial
    # coordinates for the atmosphere
    model_parameters.set_atmospheric_channel_fourier_modes(2, 2)
    # Mode truncation at the wavenumber 2 in the x and at the
    # wavenumber 4 in the y spatial coordinates for the ocean
    model_parameters.set_oceanic_basin_fourier_modes(2, 4)

    # Setting MAOOAM parameters according to the publication
    model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                                 'h': 136.5, 'd': 1.1e-7})
    model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3,
                                                      'hlambda': 15.06})
    model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})

    # Setting the short-wave radiation component: C_{a,1} and C_{o,1}
    model_parameters.atemperature_params.set_insolation(103.3333, 0)
    model_parameters.gotemperature_params.set_insolation(310, 0)

    # Build the analytic inner products (atmosphere coupled to ocean)
    # aip = AtmosphericAnalyticInnerProducts(model_parameters)
    # oip = OceanicAnalyticInnerProducts(model_parameters)
    # aip.connect_to_ocean(oip)

    # Printing the model's parameters
    # model_parameters.print_params()

    # -----------------------------------------------------------------------
    # Outputting the model equations
    # -----------------------------------------------------------------------

    # Calculating the tendencies in Python as a function of the selected parameters
    nonfixed_params = [
        model_parameters.atmospheric_params.kd,
        model_parameters.atmospheric_params.kdp,
        model_parameters.atmospheric_params.sigma,
        model_parameters.oceanic_params.gp,
        model_parameters.oceanic_params.r,
        model_parameters.oceanic_params.h,
        model_parameters.oceanic_params.d,
        model_parameters.atemperature_params.gamma,
        model_parameters.atemperature_params.C[0],
        model_parameters.atemperature_params.eps,
        model_parameters.atemperature_params.T0,
        model_parameters.atemperature_params.sc,
        model_parameters.atemperature_params.hlambda,
        model_parameters.gotemperature_params.gamma,
        model_parameters.gotemperature_params.C[0],
        model_parameters.gotemperature_params.T0,
    ]

    funcs, = create_symbolic_tendencies(
        model_parameters,
        continuation_variables=nonfixed_params,
        language='python',
    )

    # Save tendency function to a new script
    with open("tendencies_nonfixed_params.py", "w") as f_io:
        f_io.write(funcs)


# qgs computes the symbolic inner products with a multiprocessing pool. On
# macOS/Windows the start method is "spawn", which re-imports this module in
# each worker, so the executable code must be guarded by __main__.
if __name__ == '__main__':
    main()
