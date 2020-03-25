import numpy as np
import saphires as saph
import pandas as pd
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import optimize as opt
import time
from math import e
import multiprocessing as mp


def tau_IR(w, Av, Rv=3.1):
    """
    A function that returns the optical depth as a function of wavelength
    for a given visual extinction following the Cardelli et al. 1989 extinction
    curve.

    This is only relevant for wavelengths 9,090A to 33,333A.

    To extinction correct a spectrum do this:
    f_ext_corr=f_obs/(e**-tau)

    Parameters
    ----------
    w : array like
       An array of wavelengths you want the optical depth for. Units must be
       angstroms.

    Av : float
       Visual extinction value.

    Rv : float, optional
       Redding value. Default is 3.1, the galactic standard but for extinctions
       higher than Av=5, a different value might be more appropriate.

    Returns
    -------
    One Array:
       The optical depth associated with each input wavelength (in angstronms)
       for the given visual extinction.
    """

    x = 1.0 / (w * 10 ** -4)

    a = 0.574 * x ** 1.61
    b = -0.527 * x ** 1.61

    Al = (a + b / Rv) * Av

    tau = Al / 1.086

    return tau


def line_param_errs(wavelength, flux_in, err, emission_wavelength, flat_region, ext, continuum=np.nan,
                    calculation_window=800, plot=False, parallelize=False):
    mc_calcs = []
    errors = np.zeros((6, 3))
    errors[:, :] = np.nan
    if parallelize:
        pool = mp.Pool(mp.cpu_count())
        '''
        for i in range(1000):
            pool.apply_async(parallel_bootstrap_mc, args=(i, wavelength, flux_in, err, 1), callback=collect_result)
        pool.close()
        pool.join()
        mc_vals.sort(key=lambda x: x[0])
        mc_temp = [r for i, r in mc_vals]
        mc_vals_final = np.asarray(mc_temp)

        for i in range(5):
            errors[i, :] = np.percentile(mc_vals_final[:, i], (50, 68.72, 31.28))
            '''
    else:
        for i in range(100):
            mc_calcs.append(line_parameters(wavelength, (flux_in + np.random.normal(0, err, np.size(flux_in))),
                                            emission_wavelength, flat_region, ext, continuum, calculation_window,
                                            plot))
        mc_calcs.sort(key=lambda x: x[0])
        mc_calcs = np.asarray(mc_calcs)
        print(np.shape(mc_calcs))
        for i in range(6):
            errors[i, :] = np.percentile(mc_calcs[:, i], (50, 68.72, 31.28))
    return errors


def line_parameters(wavelength, flux_in, emission_wavelength, flat_region, ext, continuum=np.nan,
                    calculation_window=800, plot=False):
    """
    A function that calculates the peak flux, full width at half maximum,
    full width at zero intensity, total flux, and equivalent width of a
    given emission line.

    Parameters
    ----------
    wavelength : array-like
    Wavelength array of the spectrum to analyze, assumed to be Angstroms but
    doesn't matter as long as the default inputs are changed as well.

    flux_in : array-like
    Flux array of the spectrum to analyze.

    emission_wavelength : float
    Expected wavelength for the emission line, must match the units of the
    wavelength array.

    continuum : float
    Value of the flux at the continuum, default is nan and is then calculated
    as the median in a sigma clipping algorithm.

    calculation_window : float
    Window (in km/s), centered around 0 km/s, in which the calculations will be
    done for the given emission line.

    Returns
    -------
    peak_flux : float
    Flux value at the peak of the polynomial fit to the emission line.

    fwhm: float
    Value of the full width at half maximum of the emission line in km/s.

    fwzi: float
    Value of the full width at zero intensity of the emission line in km/s.

    total_flux: float
    Value of the flux integrated from the emission line to the line calculated
    from the endpoints of the normalized flux relative to the continuum.

    eq_width: float
    Value of the equivalent width calculated from the continuum.

    """

    # Continuum normalizes the flux using saphires and a mask around the emission line
    mask = ((wavelength < emission_wavelength - 10) | (wavelength > emission_wavelength + 10))
    try:
        if ~np.isnan(ext):
            tau = tau_IR(wave, float(ext))
            flux = flux_in / (e ** -tau)
        else:
            flux = flux_in[:]

        spline = saph.extras.bspline_acr.iterfit(wavelength[mask], flux[mask],
                                                 maxiter=15, lower=0.3, upper=0.5, nord=3,
                                                 bkspace=np.int(wavelength.size * 0.6))[0]
        flux_cont_fit = spline.value(wavelength)[0]
    except ValueError:
        flux = flux_in[:]
        flux_cont_fit = flux_in[:]

    flux_normalized = flux / flux_cont_fit

    # Converts wavelengths to velocity using doppler shift [km/s] and clips arrays within a span of
    velocity = (wavelength - emission_wavelength) * 299792. / emission_wavelength
    emission_vel_range = np.where((velocity > (0. - calculation_window/2)) & (velocity < calculation_window/2))
    velocity = velocity[emission_vel_range]

    if flat_region is None:
        flat = flux[:]
    else:
        flat = flux[np.where((wavelength > flat_region[0]) & (wavelength < flat_region[-1]))]

    for i in range(10):
        med = np.median(flat)
        std = np.std(flat)
        flat = flat[abs(flat - med) < 3.0 * std]

    if continuum is None:
        continuum = np.median(flat)
    noise = np.std(flat)

    if plot:
        fig = plt.figure(1)
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax1.plot(velocity, flux_normalized[emission_vel_range], 'r-o', ms=1.0, lw=0.75, label='Br Emission in Velocity Space')
        ax1.axvline(x=0, linestyle=':', color='black', lw=1)
        ax2.plot(wavelength, flux, 'r', lw=0.5, label='Extinction Corrected Flux')
        ax2.plot(wavelength, flux_in, 'b', lw=0.5, label='Raw Flux, $\sigma_{c}$= %0.3f' % noise)
        ax2.plot(wavelength, flux_normalized, 'g', lw=0.5, label='Normalized Flux')

    flux_normalized = flux_normalized[emission_vel_range]
    flux = flux[emission_vel_range]
    wavelength = wavelength[emission_vel_range]

    if len(flux_normalized[np.where((flux_normalized - 1) > 3 * noise)]) > 5:
        try:
            peak_range = np.where((flux_normalized - continuum > (np.max(flux_normalized) - continuum) * 0.5) &
                              (abs(velocity) < 100 + abs(velocity[np.where(flux_normalized == np.max(flux_normalized))])))
            peak_fit = np.polyfit(velocity[peak_range], flux_normalized[peak_range], 4)
            peak_fit_function = np.poly1d(peak_fit)

            peak_flux = np.max(peak_fit_function(flux_normalized[peak_range]))

            # Creates x,y array for the fit and finds the velocity value that corresponds with the peak flux value
            peak_fit_velocity = velocity[peak_range]
            peak_fit_flux = peak_fit_function(flux_normalized[peak_range])
            peak_flux_velocity = np.mean(peak_fit_velocity[np.isclose(peak_fit_flux, peak_flux)])

            half_peak_intensity = 0.5 * (peak_flux + continuum)
            amp = flux_normalized - continuum  # Establishes the amplitude of the line, sans continuum
            hpi_amp = half_peak_intensity - continuum

            hpi_range_blue = np.where((amp > hpi_amp * 0.75) & (amp < hpi_amp * 1.25) &
                                      (velocity < velocity[np.where(flux_normalized == np.max(flux_normalized))]))
            line_fit_blue = np.polyfit(flux_normalized[hpi_range_blue], velocity[hpi_range_blue], 1)
            line_blue = np.poly1d(line_fit_blue)
            half_vel_blue = line_blue(half_peak_intensity)

            hpi_range_red = np.where((amp > hpi_amp * 0.75) & (amp < hpi_amp * 1.25) &
                                     (velocity > velocity[np.where(flux_normalized == np.max(flux_normalized))]))
            line_fit_red = np.polyfit(flux_normalized[hpi_range_red], velocity[hpi_range_red], 1)
            line_red = np.poly1d(line_fit_red)
            half_vel_red = line_red(half_peak_intensity)

            fwhm = half_vel_red - half_vel_blue

            flux_offset = flux_normalized - (continuum + 0.005 * (2 * half_peak_intensity - continuum))
            end_interp = interp1d(velocity, flux_offset)

            right_y_min = np.min(flux_offset[np.where((velocity > half_vel_red))])
            right_x_min = np.min(velocity[np.where(flux_offset == right_y_min)])
            left_y_min = np.min(flux_offset[np.where((velocity < half_vel_blue))])
            left_x_min = np.min(velocity[np.where(flux_offset == left_y_min)])
            end_left_x = np.min(opt.brentq(end_interp, left_x_min, half_vel_blue))
            end_left_y = end_interp(end_left_x) + (continuum + 0.005 * (2 * half_peak_intensity - continuum))
            end_right_x = np.min(opt.brentq(end_interp, half_vel_red, right_x_min))
            end_right_y = end_interp(end_right_x) + (continuum + 0.005 * (2 * half_peak_intensity - continuum))
            fwzi = float(end_right_x) - float(end_left_x)

            endpt_range = np.where((velocity > end_left_x) & (velocity < end_right_x))
            wavelength_int = wavelength[endpt_range]
            flux_int = flux[endpt_range]
            r = (continuum - flux_int) / continuum  # R = (fc - f)/fc
            slope = (flux_int[-1] - flux_int[0]) / (wavelength_int[-1] - wavelength_int[0])
            offset = slope * (wavelength_int - wavelength_int[0]) + flux_int[0]
            total_flux = np.trapz((flux_int - offset), wavelength_int)
            eq_width = np.trapz(r, wavelength_int)
            offset_mdpt = np.median(offset)

            if plot:
                ax1.plot(velocity, flux_normalized, 'r-o', ms=1.0, lw=0.75, label='Br Emission in Velocity Space')
                print(end_left_y)
                ax1.fill_between(velocity[endpt_range], flux_normalized[endpt_range], end_left_y, color='m', alpha=0.25,
                                 label='Eq. Width = %0.3f' % eq_width)
                ax1.axvline(x=0, linestyle=':', color='black', lw=1)
                ax1.plot(peak_flux_velocity, peak_flux, 'vb', label='Fit Peak = %0.3f' % peak_flux)
                ax1.plot(half_vel_blue, half_peak_intensity, '^k', label='FWHM = %0.3f' % fwhm)
                ax1.plot(half_vel_red, half_peak_intensity, '^k')
                ax1.axhline(y=1, color='green', lw=0.5)
                ax1.plot(end_left_x, end_left_y, 'sm', label='FWZI = %0.3f' % fwzi)
                ax1.plot(end_right_x, end_right_y, 'sm')

                ax2.plot(wavelength_int, offset)

        except (RuntimeError, IndexError, ValueError, TypeError):
            peak_flux = np.nan
            fwhm = np.nan
            fwzi = np.nan
            total_flux = np.nan
            eq_width = np.nan
            offset_mdpt = np.nan
    else:
        print('No Emission Found')
        peak_flux = np.nan
        fwhm = np.nan
        fwzi = np.nan
        total_flux = np.nan
        eq_width = np.nan
        offset_mdpt = np.nan

    if plot:
        ax1.set_ylabel('Flux', fontsize=12)  # sets y axis label for flux
        ax1.set_xlabel('Velocity (km/s)', fontsize=12)
        ax1.legend()
        ax2.set_ylabel('Flux', fontsize=12)
        ax2.legend()
        plt.show()
        fig.clf()
    calcs = (peak_flux, fwhm, fwzi, total_flux, eq_width, offset_mdpt)
    return calcs

