import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import minimize, curve_fit
from scipy.stats import mode
from scipy.signal import savgol_filter, find_peaks, peak_widths, boxcar
from numpy.random import normal
from astropy.convolution import convolve as astropy_convolve

from statsmodels.robust import mad

def create_datafilename(sol, filename_stem="_calib_", dr='/Users/brian/Downloads/ps_bundle/data_calibrated'):
    filenames = glob.glob("%s/*/*%s*.csv" % (dr, filename_stem))
    
    # file names have some zeros and then the sol number; Add the right number of zeros
    file_stem = ""
    for i in range(4 - len(str(int(sol)))):
        file_stem += "0"
    file_stem += str(int(sol))
    
    res = [i for i in filenames if str(file_stem) + "_0" in i] 
    return res

def convert_ltst(sol_data):
    LTST = np.zeros(sol_data['LTST'].size)
    LTST_and_sol = np.zeros(sol_data['LTST'].size)
    
    for i in range(len(sol_data['LTST'])):
        sol, time = sol_data['LTST'][i].decode("utf-8").split()
        hr, mn, sc = time.split(":")
        
        cur_LTST = float(hr) + float(mn)/60. + float(sc)/3600.
        LTST[i] = cur_LTST
        LTST_and_sol[i] = float(sol)*24. + cur_LTST
        
    return LTST, LTST_and_sol

def retrieve_vortices(sol, sol_filename, Spigas_data, window_width=30./3600, subtract_time=True):
    sol_data = np.genfromtxt(sol_filename[0], delimiter=",", dtype=None, names=True)
    LTST, LTST_and_sol = convert_ltst(sol_data)

    vortices = []
    central_times = []
    
    ind = Spigas_data['SOL'] == sol
    for time in Spigas_data['_LTST_'][ind]:
        x = LTST
        time_ind = np.abs(x - time) < window_width
        if(subtract_time):
            x = LTST - time
            time_ind = np.abs(x) < window_width
            
        vortices.append([x[time_ind], sol_data['PRESSURE'][time_ind]])
        central_times.append(time)
                                           
    return vortices

def central_times(sol, sol_filename, Spigas_data):
    ind = Spigas_data['SOL'] == sol
    return Spigas_data['_LTST_'][ind]

def modified_lorentzian(t, baseline, slope, t0, DeltaP, Gamma):
    # Equation 7 from Kahapaa+ (2016)
    return baseline + slope*(t - t0) - DeltaP/(((t - t0)/(Gamma/2.))**2 + 1)

# From https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(x, w, mode='valid'):
    return np.convolve(x, np.ones(w), mode) / w

def moving_std(x, w, mode='valid'):
    avg = moving_average(x, w, mode=mode)
    return np.sqrt(w/(w - 1.)*moving_average((x - avg)**2, w, mode=mode))

def redchisqg(ydata,ymod,deg=2,sd=None):
    """  
    Returns the reduced chi-square error statistic for an arbitrary model, 
    chisq/nu, where nu is the number of degrees of freedom. If individual   
    standard deviations (array sd) are supplied, then the chi-square error
    statistic is computed as the sum of squared errors divided by the standard   
    deviations. See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.
    
    ydata,ymod,sd assumed to be Numpy arrays. deg integer.
    
    Usage:
    chisq=redchisqg(ydata,ymod,n,sd)
    
    where  
    ydata : data  
    ymod : model evaluated at the same x points as ydata
    n : number of free parameters in the model  
    sd : uncertainties in ydata
    
    Rodrigo Nemmen
    http://goo.gl/8S1Oo
    """
    # Chi-square statistic
    if(np.any(sd == None)):
        chisq=np.sum((ydata-ymod)**2)
    else:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )
        
    # Number of degrees of freedom assuming 2 free parameters
    nu=ydata.size - 1. - deg
    
    return chisq/nu       

def condition_vortex(vortex):
    
    _, unq = np.unique(vortex[0], return_index=True)            
    x = vortex[0][unq]
    y = vortex[1][unq]
    
    # Remove NaNs
    ind = np.isfinite(x) & np.isfinite(y)
    x = x[ind]
    y = y[ind]
    
    return x, y

def fit_vortex(vortex, init_params, bounds, rescale_uncertainties=True, zoomed_in=None):
    
    x, y = condition_vortex(vortex)
    
    if(zoomed_in is not None):
        ind = np.abs(x - init_params[2]) < zoomed_in
        x = x[ind]
        y = y[ind]
    
    # First fit out the long-term slope
    fit_params = np.polyfit(x, y, 1)
    detrended_data = y - np.polyval(fit_params, x)
                    
    popt, pcov = curve_fit(modified_lorentzian, x, y, p0=init_params, bounds=bounds)
    ymod = modified_lorentzian(x, *popt)
    
    if(rescale_uncertainties):       
        sd = mad(y - ymod)
        red_chisq = redchisqg(y, ymod, deg=5, sd=sd)
        
        pcov *= np.sqrt(red_chisq)
        
    return popt, np.sqrt(np.diag(pcov))

def determine_init_params(vortex, 
                          init_baseline=None, init_slope=None, init_t0=None, init_DeltaP=None, init_Gamma=None):

    x, y = condition_vortex(vortex) 
    fit_params = np.polyfit(x, y, 1)
    detrended_y = y - np.polyval(fit_params, x)
    
    if(init_baseline is None):
        init_baseline = np.median(y)
        
    if(init_slope is None):
        init_slope = fit_params[0]
        
    if(init_t0 is None):
        init_t0 = x[np.argmin(detrended_y)]

    if(init_DeltaP is None):
        init_DeltaP = 10.
        
    if(init_Gamma is None):
        init_Gamma = 2./3600.
        
    return np.array([init_baseline, init_slope, init_t0, init_DeltaP, init_Gamma])

def determine_bounds(vortex, init_params, init_t0_fac=0.0002, init_DeltaP_fac=1000.):
    
    x, y = condition_vortex(vortex)
    
    return ([np.min(y), -10.*np.abs(init_params[1]), (1-init_t0_fac)*init_params[2], 0., 0.5/3600], 
            [np.max(y),  10.*np.abs(init_params[1]), (1+init_t0_fac)*init_params[2], init_DeltaP_fac*init_params[3],
               300./3600])

def retrieve_data(sol, filename_stem="calib", dr=None, nans_in_gaps=False, data_field="PRESSURE"):
    sol_filename = create_datafilename(sol, filename_stem=filename_stem, dr=dr)

    sol_data = np.genfromtxt(sol_filename[0], delimiter=",", dtype=None, names=True)
    LTST, LTST_and_sol = convert_ltst(sol_data)

    if(data_field is not None):
        ind = np.isfinite(LTST) & np.isfinite(LTST_and_sol) &\
                np.isfinite(sol_data[data_field])
        LTST = LTST[ind]
        LTST_and_sol = LTST_and_sol[ind]
        sol_data = sol_data[ind]

    # For some reason, some times are doubled up
    _, unq = np.unique(sol_data["LTST"], return_index=True)
    LTST = LTST[unq]
    LTST_and_sol = LTST_and_sol[unq]
    sol_data = sol_data[unq]

    if(nans_in_gaps is True):
        LTST, LTST_and_sol, sol_data = fill_gaps(LTST, LTST_and_sol, sol_data)

    return LTST, LTST_and_sol, sol_data

def fill_gaps(LTST, LTST_and_sol, sol_data):
    """ Fill gaps in the time-series """

    ret_LTST = LTST
    ret_LTST_and_sol = LTST_and_sol
    ret_sol_data = sol_data

    gaps = find_gaps(LTST_and_sol)

    delta_ts = (LTST_and_sol[1:] - LTST_and_sol[0:-1])
    ind = delta_ts > 0.
    mod = mode(delta_ts[ind])[0][0]

    if(len(gaps) == 0):
        return LTST, LTST_and_sol, sol_data
    else:
        ret_LTST_and_sol = LTST_and_sol
        ret_LTST = LTST
        ret_sol_data = sol_data

        for cur_gap in gaps:
            ret_LTST_and_sol = np.append(ret_LTST_and_sol,
                    np.arange(LTST_and_sol[cur_gap] + mod,
                        LTST_and_sol[cur_gap+1] - mod, mod))
    
            ret_LTST = np.append(ret_LTST, np.arange(LTST[cur_gap] + mod,
                LTST[cur_gap+1] - mod, mod))

            # Create temp array to get correct shape
            temp_array = np.empty(len(np.full_like(np.arange(LTST[cur_gap] + mod,
                            LTST[cur_gap+1] - mod, mod), np.nan)),
                            dtype=sol_data.dtype)
            for cur_name in sol_data.dtype.names:
                temp_array[cur_name] = np.full_like(np.arange(LTST[cur_gap] + mod,
                            LTST[cur_gap+1] - mod, mod), np.nan)
            ret_sol_data = np.append(ret_sol_data, temp_array)

        srt = np.argsort(ret_LTST_and_sol)
        ret_LTST_and_sol = ret_LTST_and_sol[srt]
        ret_LTST = ret_LTST[srt]
        ret_sol_data = ret_sol_data[srt]

        return ret_LTST, ret_LTST_and_sol, ret_sol_data

def break_at_gaps(LTST, LTST_and_sol, sol_data):
    """ Break the time-series into pieces if there are gaps """

    gaps = find_gaps(LTST_and_sol)

    if(len(gaps) == 0):
        return [LTST], [LTST_and_sol], [sol_data]
    else:
        ret_LTST = list()
        ret_LTST_and_sol = list()
        ret_sol_data = list()

        last_gap = 0
        for i in range(len(gaps)):

            ret_LTST.append(LTST[last_gap:gaps[i]+1])
            ret_LTST_and_sol.append(LTST_and_sol[last_gap:gaps[i]+1])
            ret_sol_data.append(sol_data[last_gap:gaps[i]+1])

            last_gap = gaps[i]

        ret_LTST.append(LTST[gaps[-1]+1:])
        ret_LTST_and_sol.append(LTST_and_sol[gaps[-1]+1:])
        ret_sol_data.append(sol_data[gaps[-1]+1:])

        return ret_LTST, ret_LTST_and_sol, ret_sol_data

def find_gaps(LTST_and_sol):
    """ Finds gaps in the time-series """

    delta_ts = (LTST_and_sol[1:] - LTST_and_sol[0:-1])
    ind = delta_ts > 0.
    mod = mode(delta_ts[ind])[0][0]

    # If there are no gaps in the time-series
    if(len(delta_ts[~np.isclose(delta_ts, mod)]) == 0):
        return []
    else:
        return np.argwhere(~np.isclose(delta_ts, mod))[:,0]

def boxcar_filter(LTST, LTST_and_sol, sol_data, boxcar_window_size):

    gapped_LTST, gapped_LTST_and_sol, gapped_sol_data = break_at_gaps(LTST, LTST_and_sol, sol_data)

    filt = np.array([])
    st = np.array([])
    for cur in gapped_sol_data:
        
        filt = np.append(filt, astropy_convolve(cur["PRESSURE"], boxcar(boxcar_window_size),
            boundary='extend', preserve_nan=True))
        st = np.append(st, moving_std(cur["PRESSURE"], boxcar_window_size, mode='same'))

    # Yank the NaNs
#   ind = ~np.isnan(filt)
#   filt = filt[ind]
#   st = st[ind]

    return filt, st

def apply_lorentzian_matched_filter(time, filtered_data, st, lorentzian_fwhm, lorentzian_depth, delta_t=None):

    if(delta_t is None):
        delta_t = np.max(time[1:] - time[0:-1])

    lorentzian_time = np.arange(-3.*lorentzian_fwhm, 3.*lorentzian_fwhm, delta_t)
    lorentzian = modified_lorentzian(lorentzian_time, 0., 0., 0., lorentzian_depth, lorentzian_fwhm)

    convolution = np.convolve(filtered_data/st, lorentzian, mode='same')
#   convolution -= np.median(convolution)
#   convolution /= mad(convolution)

    return convolution

def find_vortices(time, convolution, detection_threshold=5):
    """Finds outliers """

    med = np.median(convolution)
    md = mad(convolution)

    convolution -= med
    convolution /= md

    ex = find_peaks(convolution)
    ind = convolution[ex[0]] >= detection_threshold

    pk_wds, _, _, _ = peak_widths(convolution, ex[0][ind])

    return np.searchsorted(time, time[ex[0]][ind]), pk_wds

def line(x, m, b):
    return m*x + b

def find_wind(cur_sol, t0, Gamma, filename_stem="_model_", num_Gamma=10.):
    # Wind data in a different folder
    dr_wind = '/Users/bjackson/Downloads/twins_bundle/data_derived'

    try:
        wind_LTST, wind_LTST_and_sol, wind_data =\
            retrieve_data(cur_sol, dr=dr_wind, 
                    filename_stem=filename_stem, 
                    data_field="HORIZONTAL_WIND_SPEED")
        return wind_LTST, wind_LTST_and_sol, wind_data

    except:
        print("%s doesn't have windspeed data!" % cur_sol)
        return None, None, None

def estimate_diameter(sol, t0, Gamma, Gamma_err, 
        num_max_gam=5., num_min_gam=3.):
    wind_LTST, wind_LTST_and_sol, wind_data = find_wind(sol, t0, Gamma)
    wind_LTST_and_sol -= 24.*sol

    ind = (np.abs(wind_LTST_and_sol - t0)*3600. < num_max_gam*Gamma) &\
        np.abs((wind_LTST_and_sol - t0)*3600. > num_min_gam*Gamma)

    if(len(wind_data["HORIZONTAL_WIND_SPEED"][ind]) > 0):
        med = np.median(wind_data["HORIZONTAL_WIND_SPEED"][ind])
        md = np.nan

        diameter = med*Gamma
        diameter_unc = np.nan
        if(len(wind_data["HORIZONTAL_WIND_SPEED"][ind]) > 1):
            md = mad(wind_data["HORIZONTAL_WIND_SPEED"][ind])
            diameter_unc = diameter*np.sqrt((md/med)**2 + (Gamma_err/Gamma)**2)

        return diameter, diameter_unc, med, md

    else:
        return np.nan, np.nan, np.nan, np.nan

def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# Define a function (quadratic in our case) to fit the data with.
def lin_func(p, x):
    m, c = p
    return m*x + c

def velocity_profile(t, t0, Gamma, VT, background_wind):
    """ Returns wind velocity profile """

    return VT*(t - t0)/(Gamma/2.)/(1. + ((t - t0)/(Gamma/2.))**2) +\
            background_wind
