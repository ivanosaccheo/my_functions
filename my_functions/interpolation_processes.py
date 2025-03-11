from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.interpolate import interp1d

import numpy as np


def get_GPR(luminosity, grid, kernel = ConstantKernel()* Matern(10, (1e-5,1e5), nu =0.5),  extrapolate = False, wav_min = 912):
    
    isnan = np.isnan(luminosity[:,1])
    wav_min = np.log10(wav_min)
    
    x = luminosity[~isnan,0]
    y = luminosity[~isnan,1]
    dy = luminosity[~isnan,2]
    
    y = y[x>=wav_min]
    dy = dy[x>=wav_min]
    x = x[x>=wav_min]
    

    
    if extrapolate: 
        logico = np.ones(len(grid), dtype=bool)
    else:
        logico = np.logical_and(grid >=x[0], grid <=x[-1])  
    
    interpolated_data = np.empty((len(grid),3))
    interpolated_data.fill(np.nan)
    interpolated_data[:,0] = grid
    
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=dy**2, n_restarts_optimizer=10)
    gpr.fit(x[:, None], y)
    interpolated_data[logico,1], interpolated_data[logico,2] = gpr.predict(grid[logico,None] , return_std=True)
    
    return interpolated_data

def get_linear_interpolation(luminosity, grid, extrapolate=False, wav_min = 912, montecarlo = False):
    
    isnan = np.isnan(luminosity[:,1])
    wav_min = np.log10(wav_min)
    
    x = luminosity[~isnan,0]
    y = luminosity[~isnan,1]
    dy = luminosity[~isnan,2]
    
    y = y[x>=wav_min]
    dy = dy[x>=wav_min]
    x = x[x>=wav_min]
    
    
    
    if montecarlo:
        y = np.random.normal(loc=y, scale=dy)

    interpolated_data = np.empty((len(grid),3))
    interpolated_data.fill(np.nan)
    interpolated_data[:,0] = grid


    if not extrapolate:
        
        interpolated_data[:,1] = np.interp(grid, x, y, left = np.nan, right = np.nan)
        interpolated_data[:,2] = np.interp(grid, x, y+dy, left = np.nan, right = np.nan)
        interpolated_data[:,2] = interpolated_data[:,2]-interpolated_data[:,1]

    else:  

        func_y = interp1d(x, y, fill_value= "extrapolate")
        func_dy = interp1d(x, y+dy, fill_value= "extrapolate")
        interpolated_data[:,1] = func_y(grid)
        interpolated_data[:,2] = func_dy(grid)-interpolated_data[:,1]
    
    return interpolated_data


