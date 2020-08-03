import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import minimize, curve_fit
from scipy.stats import mode
from scipy.signal import argrelextrema
from numpy.random import normal

from statsmodels.robust import mad

def create_datafilename(sol, dr='/Users/brian/Downloads/ps_bundle/data_calibrated'):
    filenames = glob.glob(dr + "/*/*_calib_*.csv")
    
    # file names have some zeros and then the sol number; Add the right number of zeros
    file_stem = ""
    for i in range(4 - len(str(sol))):
        file_stem += "0"
    file_stem += str(sol)
    
    res = [i for i in filenames if str(file_stem) + "_01" in i] 
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
    if(sd==None):
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

def retrieve_pressure_data(sol, dr=None):
    sol_filename = create_datafilename(sol, dr=dr)

    sol_data = np.genfromtxt(sol_filename[0], delimiter=",", dtype=None, names=True)
    LTST, LTST_and_sol = convert_ltst(sol_data)

    ind = np.isfinite(LTST) & np.isfinite(LTST_and_sol) &\
            np.isfinite(sol_data["PRESSURE"])
    LTST = LTST[ind]
    LTST_and_sol = LTST_and_sol[ind]
    sol_data = sol_data[ind]

    # For some reason, some times are doubled up
    _, unq = np.unique(sol_data["LTST"], return_index=True)
    LTST = LTST[unq]
    LTST_and_sol = LTST_and_sol[unq]
    sol_data = sol_data[unq]

    return LTST, LTST_and_sol, sol_data
