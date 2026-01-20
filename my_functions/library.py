##### Utili 
# interpolate    equal to np.interp :(
# two_2_three    transform tables into 3d array
# three_2_two    inverse of two_2_three
# compute_mean_in_bins   compute the mean and variance in the provided bins



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os


PATH_TO_DATA =  os.path.expanduser("~/WORK/my_functions/")




def interpolate(x, y, x0, out_of_bounds = "extrapolate", sort= False, log_log=True):
    """
    Just calls intep1d, kept for legacy

    """
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    if sort:  # sorting the templates
       order = np.argsort(x)
       x, y = x[order], y[order]
    
    if log_log:
        x, y = np.log10(x), np.log10(y)
        x0 = np.log10(x0)
    bounds_error = False
    
    if out_of_bounds == "error":
        bounds_error=True
        out_of_bounds = 0
    f = interp1d(x, y, fill_value = out_of_bounds, bounds_error=bounds_error)
    y0 = f(x0)
    if log_log:
        y0 = 10**y0
    return y0
       
    

def two_2_three(data, extra_features = False, has_wavelength=True):
    """
    It converts a 2-Dimensional table into a 3-D numpy array. 
    The last columns of the table are removed and saved as an independent 2D table.
    The number of columns to be remvoded is provided by the variable 'features'. It can be
    both an integer number or a list of strings containing the names of the columns to 
    be removed. The list MUST contain all other data except from photometry or luminosities.

    Parameters
    ----------
    data : Pandas Data-Frame or Numpy 2D array
           Table with magnitudes/luminosities + other features (e.g. redshift).
           
    extra_features : List of strings or integers
               Defines the other features in the table. If the original table is 
               (lambda_u, u, err_u, redshift, EBV), then features can be passed both as 
               [3,4] or ['redshift', 'EBV']
    has_Wavelength : Logical
               Whether the table has the wavelengths or not 
    
    Returns
    NewData = numpy 3D array 
    
    other_features = 2D numpy array/ pandas Dataframe containing the extra features extracted from the original 
                     table
    """
    if isinstance(data, pd.core.frame.DataFrame):     #if pandas
        if extra_features:
            assert isinstance(extra_features, list)
            
            if all(isinstance(i, int) for i in extra_features):
                col_to_strip =[data.columns[i] for i in extra_features]
            elif all(isinstance(i, str) for i in extra_features):
                for name in extra_features: 
                    assert(name in data.columns)
                col_to_strip = extra_features
            else: 
                raise Exception('La lista di feature deve essere composta o solo da stringhe o da interi')
            
            other_features = data[col_to_strip]
            raw_features = [c for c in data.columns if c not in col_to_strip]
            raw_data = data[raw_features].to_numpy()

        else:
            raw_data = data.to_numpy() 

    elif isinstance(data, np.ndarray):
        if extra_features:
            assert isinstance(extra_features, list)
            assert all(isinstance(i, int) for i in extra_features)
            other_features = data[:,extra_features]
            raw_features = [i for i in range(data.shape[1]) if i not in extra_features]
            raw_data = data[:,raw_features]
        else:
            raw_data = data
    else:
        raise Exception('Data ust be a Pandas DataFrame or a numpy 2D array')
     
    Nqso = raw_data.shape[0]
    if has_wavelength:  
        Nproperties = 3
    else:
        Nproperties = 2
    Nbands = raw_data.shape[1] // Nproperties
    NewData = raw_data.reshape(Nqso, Nbands, Nproperties)
    #NewData =np.zeros((Nqso, Nbands, Nproperties))

    #for i in range(Nbands):
    #        for k in range(Nproperties):
    #            NewData[:,i,k] =raw_data[:,i*Nproperties+k]      
    #           
    if extra_features:
       return NewData, other_features
    
    return NewData

def three_2_two(data, *args, band_names = None, all_names = None ):
    """
    It transforms a 3-Dimesional array into a 2D table, where each row has data from 1 source.
    Table's columns give luminosities at the different bands + possibly other physical information (e.g. redshift)

    Parameters
    ----------
    data : Numpy 3D array.
           Numpy 3D array with magnitudes or luminosity
    *args : other_data i.e. redshift, EBV, Lbol to include  where each feature is a 
                 NQSO x 1 array. 
       
    band_names : List of strings, OPTIONAL
            list of strings, containig the names of the bands ( e.g. u, g, K, W1) 
            and the names of the other features (e.g. redshift, EBV). 
            If passed, a Pandas Data Frame with columns names is returned
    
    all_names : List of strings, OPTIONAL
            list of strings, containig the all the names (including err_ or lambda_)
            if all_names and band_names are provided, all_names is used
            

    Returns : Pandas Data Frame
    """
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("data must be a 3D numpy array")

    Nqso, Nbands, Ndata = data.shape
    flat_data = data.reshape(Nqso, Nbands * Ndata)

    if args:
        for feature in args:
            if len(feature) != Nqso:
                raise ValueError("All extra features must have length Nqso")

        extras = np.column_stack(args)
        NewData = np.hstack([flat_data, extras])
    else:
        NewData = flat_data
    
    name_array = None
    if all_names:
        name_array = all_names
    
    elif  band_names:
        name_array =[]
        for i, name in enumerate(band_names):
            if i < Nbands:
                if Ndata == 3:
                    name_array += [f"lambda_{name}", name, f"err_{name}"]
                elif Ndata == 2:
                    name_array += [name, f"err_{name}"]
            #Extra features (redshift, EBV, etc)
            else:
                name_array.append(name) 
    return pd.DataFrame(NewData, columns = name_array)  

def add_wavelength(magnitudes, wavelen):
    assert(len(wavelen)== magnitudes.shape[1])
    new_magnitudes = np.zeros((magnitudes.shape[0],magnitudes.shape[1], 3))
    new_magnitudes[:,:,0] = wavelen
    new_magnitudes[:,:,1:]=magnitudes
    return new_magnitudes


def my_binned_statistic(x, y, 
                        func,
                        bins = 10, 
                        include_counts=False
                        ):
    #same as scipy.stats.binned_statistic, but func can return also non scalar
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isscalar(bins):
        bins = np.linspace(x.min(), x.max(), bins+1)
    else:
        bins = np.asarray(bins)
    
    Nbins = len(bins) - 1
    bin_idx = np.digitize(x, bins = bins) - 1
    values = [None for _ in range(Nbins)]
    counts = np.zeros(Nbins, dtype=int)              
    for i in range(Nbins):
        select = bin_idx == i
        if np.any(select):
            values[i] = func(y[select])
            counts[i] = np.sum(select)
        else:
            values[i] = func(np.full(y.shape, np.nan))        
    if include_counts:
        return bins, np.asarray(values), counts
    else:
        return bins, np.asarray(values)
    

def my_binned_statistic_2d(x, y, z, func, 
                           xbins =10, ybins =10,
                           include_counts=False,
                           ):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if np.isscalar(xbins):
        xedges = np.linspace(x.min(), x.max(), xbins+1)
    else:
        xedges = np.asarray(xbins, dtype=float)

    if np.isscalar(ybins):
        yedges = np.linspace(y.min(), y.max(), ybins+1)
    else:
        yedges = np.asarray(ybins, dtype=float)

    Nx, Ny = len(xedges)-1, len(yedges)-1

    xi = np.digitize(x, bins=xedges) - 1
    yi = np.digitize(y, bins=yedges) - 1

    values = [[None for _ in range(Ny)] for _ in range(Nx)]
    counts = np.zeros((Nx, Ny), dtype=int)

    for i in range(Nx):
        for j in range(Ny):
            select= (xi == i) & (yi == j)
            if np.any(select):
                values[i][j] = func(z[select])
                counts[i, j] = np.sum(select)
            else:
                values[i][j] = func(np.full(z.shape, np.nan))

    if include_counts:
        return xedges, yedges, np.asarray(values), counts
    else:
        return xedges, yedges, np.asarray(values)


def get_binned_quantiles(x, y, 
                         bins = 10, 
                         quantiles = [0.05, 0.16, 0.5, 0.84, 0.95],
                         include_counts=False):
    
    quantile_func = lambda x:  np.quantile(x, quantiles)
    return my_binned_statistic(x, y, func=quantile_func,
                               bins = bins,
                               include_counts=include_counts)







def interpolate_legacy(x0, x, y, out_of_bounds= 'error', sort= True, log_log=True):
    """
    Kept 
    It returns the value of y computed at x0 linearly interpolating between 
    two adjacent points. x and y must have the same size.
    x :  N*1 array
    y :  N*1 array
    x0 : float
    out_of_bounds : Number, np.nan, 'extrapolate', 'error'. It determines the behaviour of the interpolation when x0 is out of bounds

    sort: logical, if true the templates are sorted before computation

    log_log : logical, if true it interpolates in the log-log space

    """
    x  = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    if sort:  # sorting the templates
       order = np.argsort(x)
       x, y = x[order], y[order]
    
    N = len(x)-1

    if x0 < x[0] and out_of_bounds == 'error':
        raise Exception("Value out of range, lambda too short")
    elif x0 < x[0] and out_of_bounds == 'extrapolate':
        if log_log:
            y0 = np.log(y[0])+((np.log(y[1])-np.log(y[0])) /
                               (np.log(x[1])-np.log(x[0])))*(np.log(x0)-np.log(x[0]))
            y0 = np.exp(y0)
        else:
            y0 = y[0]+((y[1]-y[0])/(x[1]-x[0]))*(x0-x[0])
    elif x0 < x[0]:
        y0 = out_of_bounds
    elif x0 > x[N] and out_of_bounds == 'error':
        raise Exception("Value out of range, lambda too long")
    elif x0 > x[N] and out_of_bounds == 'extrapolate':
        if log_log:
            y0 = np.log(y[N-1])+((np.log(y[N])-np.log(y[N-1])) /
                                 (np.log(x[N])-np.log(x[N-1])))*(np.log(x0)-np.log(x[N-1]))
            y0 = np.exp(y0)
        else:
            y0 = y[N-1]+((y[N]-y[N-1])/(x[N]-x[N-1]))*(x0-x[N-1])
    elif x0 > x[N]:
        y0 = out_of_bounds

    else:
        hi = len(x)-1  # high index
        li = 0  # low index
        while True:
            if hi-li == 1 or hi-li == 0:
                if log_log:
                    y0 = np.log(y[hi-1])+((np.log(y[hi])-np.log(y[hi-1])) /
                                          (np.log(x[hi])-np.log(x[hi-1])))*(np.log(x0)-np.log(x[hi-1]))
                    y0 = np.exp(y0)
                else:
                    y0 = y[hi-1]+((y[hi]-y[hi-1])/(x[hi]-x[hi-1]))*(x0-x[hi-1])
                break
            mi = int((hi+li)/2)  # middle index
            if x0 < x[mi]:
                hi = mi
            elif x0 >= x[mi]:
                li = mi

    return y0




   
    
   
  




















  







































  



















