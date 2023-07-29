import numpy as np
from scipy import integrate
import numpy.polynomial.polynomial as poly
from  my_functions import library as lb
import pandas as pd
from scipy.interpolate import interp1d


def lyman_continuum_LAF(redshift, lambda_obs):
    
    ll = 911.8                   #lyman-limit
    wav = lambda_obs/ll
    tau = np.zeros((len(lambda_obs),))
    
    if redshift < 1.2:
        idx = wav<(redshift+1)
        tau[idx] = 0.325*(wav[idx]**1.2-((1+redshift)**(-0.9))*(wav[idx]**2.1))
    
    elif redshift >= 1.2 and redshift < 4.7:
        idx1 = wav < 2.2
        idx2 = np.logical_and(wav >= 2.2, wav <(redshift+1))
        
        tau[idx1] = (0.0255*((1+redshift)**1.6)*(wav[idx1]**2.1) 
                           +0.325*(wav[idx1]**1.2) -0.250*(wav[idx1]**2.1))
           
        tau[idx2] = 0.0255*(((1+redshift)**1.6)*(wav[idx2]**2.1)-(wav[idx2]**3.7))
    else:
        idx1 = wav < 2.2
        idx2 = np.logical_and(wav >= 2.2, wav <5.7)
        idx3 = np.logical_and(wav >= 5.7, wav < (redshift+1))
        
        tau[idx1] = (0.000522*((1+redshift)**3.4)*(wav[idx1]**2.1) 
                     + 0.325*(wav[idx1]**1.2) - 0.0314*(wav[idx1]**2.1))
        
        tau[idx2] =  (0.000522*((1+redshift)**3.4)*(wav[idx2]**2.1) 
                      +0.218*(wav[idx2]**2.1) -0.0255*(wav[idx2]**3.7))            
            
        tau[idx3] = 0.000522*(((1+redshift)**3.4)*(wav[idx3]**2.1)- (wav[idx3]**5.5))
​
    return tau
​
​
def lyman_continuum_DLA(redshift, lambda_obs):
    
    ll = 911.8                   #lyman-limit
    wav = lambda_obs/ll
    tau = np.zeros((len(lambda_obs),))
    
    if redshift < 2:
        idx = wav < (1+redshift) 
        tau[idx] = 0.211*((1+redshift)**2) - 0.0766*((1+redshift)**2.3)*(wav[idx]**(-0.3))-0.135*(wav[idx]**2) 
​
    else:
        idx1 = wav < 3
        idx2 = np.logical_and(wav >=3, wav < (1+redshift))
   
        tau[idx1] = (0.634 + 0.047*((1+redshift)**3) -0.0178*((1+redshift)**3.3)*(wav[idx1]**(-0.3))
                    -0.135*(wav[idx1]**2)-0.291*(wav[idx1]**(-0.3)))
           
        tau[idx2] = 0.047*((1+redshift)**3)-0.0178*((1+redshift)**3.3)*(wav[idx2]**(-0.3))-0.0292*(wav[idx2]**3)
    
    return tau
    
​
def lyman_series_LAF(redshift, lambda_obs, coefficients):
    
    wav = lambda_obs ##just for clarity
    tau = np.zeros((len(lambda_obs), coefficients.shape[0]))
    
    for j in range(coefficients.shape[0]):
    
        idx1 = np.logical_and.reduce([wav < coefficients[j,1]*2.2, 
                                     wav > coefficients[j,1], 
                                     wav < coefficients[j,1]*(redshift+1)], axis = 0)
        
        idx2 = np.logical_and.reduce([wav >= coefficients[j,1]*2.2, 
                                      wav < coefficients[j,1]*5.7, 
                                      wav < coefficients[j,1]*(redshift+1)], axis = 0)
        idx3 = np.logical_and.reduce([~np.logical_or(idx1, idx2),
                                      wav > coefficients[j,1], 
                                      wav < coefficients[j,1]*(redshift+1)], axis = 0)
        
        tau[idx1, j] = coefficients[j,2]*((wav[idx1]/coefficients[j,1])**1.2)
        
        tau[idx2, j] = coefficients[j,3]*((wav[idx2]/coefficients[j,1])**3.7)
    
        tau[idx3, j] = coefficients[j,4]*((wav[idx3]/coefficients[j,1])**5.5)
    
    return np.sum(tau, axis = 1)
    
    
def lyman_series_DLA(redshift, lambda_obs, coefficients):
    
    wav = lambda_obs ##just for clarity
    tau = np.zeros((len(lambda_obs), coefficients.shape[0]))
    
    for j in range(coefficients.shape[0]):
        
        idx1 = np.logical_and.reduce([wav < coefficients[j,1]*3, 
                                      wav > coefficients[j,1],
                                      wav < coefficients[j,1]*(redshift+1)], axis = 0)
        idx2 = np.logical_and.reduce([~idx1, 
                                      wav >coefficients[j,1],
                                      wav < coefficients[j,1]*(redshift+1)], axis = 0)
    
            
        tau[idx1, j] = coefficients[j,5]*((wav[idx1]/coefficients[j,1])**2)
               
        tau[idx2, j] = coefficients[j,6]*((wav[idx2]/coefficients[j,1])**3)
              
           
    return np.sum(tau, axis = 1)
    
    
def get_IGM_absorption(redshift, lambda_obs, coefficients = get_lyman_coefficients(), DLA = True):
     """ Optical depth computed according to Inoue et al. 2014
    """
    tau_continuum_laf = lyman_continuum_LAF(redshift, lambda_obs)
    tau_series_laf = lyman_series_LAF(redshift, lambda_obs, coefficients)
    if not DLA:
         return  tau_continuum_laf + tau_series_laf 
    else:  
        tau_continuum_dla = lyman_continuum_DLA(redshift, lambda_obs)
        tau_series_dla = lyman_series_DLA(redshift, lambda_obs, coefficients)
        return  tau_continuum_laf +  tau_continuum_dla + tau_series_laf + tau_series_dla


def get_lyman_coefficients(coefficients_path= "tables/lyman_series_coefficients.dat")
    return np.loadtxt(coefficient_path)

    
def correct_magnitudes(redshift, filter_path, emission_lines = True, IGM = True, DLA = True, spectrum_path = '~/DATA/tables/vanden_berk_13.dat'):
    '''
    It returns an array with the magnitude corrections.
    It requires an array with sources redshift and an array with the path to the filter transmission (s) file.
    '''
    lambda_obs = np.loadtxt(filter_path)
    transmission = lambda_obs[:,1]
    lambda_obs = lambda_obs[:,0]
    spectrum_rest = np.loadtxt(spectrum_path, skiprows = 0) # 0 col= rest frame wav, 1 col = no Emission Lines, 2 col =with EL
    delta_M =[]
    for z in redshift:    
        delta_m_IGM =0
        delta_m_EL =0
        continuum, lines = shift_to_observed(spectrum_rest, z, lambda_obs)
        den = integrate.trapezoid(continuum*lambda_obs*transmission, lambda_obs)
        if IGM:
           tau = np.array(optical_depth(z, lambda_obs, DLA=DLA))
           y =1/np.exp(tau.astype(float))
           num = integrate.trapezoid(y*continuum*lambda_obs*transmission, lambda_obs)
           delta_m_IGM = -2.5*np.log10(num/den)
        if emission_lines:
           num = integrate.trapezoid(lines*lambda_obs*transmission, lambda_obs)
           delta_m_EL = -2.5*np.log10(num/den)
        delta_M.append(delta_m_IGM+delta_m_EL)
    delta_M = np.asarray(delta_M)
    return delta_M
    
    
def shift_to_observed(spectrum, redshift, lambda_obs):
   x = spectrum[:,0]*(redshift +1)
   continuum =[]
   lines =[]
   for wav in lambda_obs:
       y = lb.interpolate(x, spectrum[:,1], wav, out_of_bounds =0, sort =False)
       continuum.append(y)
       y = lb.interpolate(x, spectrum[:,2], wav, out_of_bounds =0, sort =False)
       lines.append(y)
   return continuum, lines
        
        
def gap_filling(magnitudes, redshift,coefficients, SED_path = '~/DATA/tables/gap_filling_sed.dat'):
    sed = np.loadtxt(SED_path, skiprows = 0) #lambda, L_lambda
    N_bands = np.shape(magnitudes)[1]
    N_qso = np.shape(magnitudes)[0]
    for j in range(N_qso):
          M = np.copy(magnitudes[j,:,:])    #find the nearest available filter without gap-filled data
          for i in range(N_bands):
               if not magnitudes[j,i,1]==magnitudes[j,i,1]:       #looking for nan values
                  nearest =find_nearest_filter(M,i)
                  Fnu = lb.interpolate(sed[:,0], sed[:,1], magnitudes[j,nearest,0]/(redshift[j]+1), sort=False, out_of_bounds =np.nan)*magnitudes[j,nearest,0]/2.998e18
                  A = 10**((-(magnitudes[j,nearest,1]+48.6)/2.5)-np.log10(Fnu))
                  Fnu_gap= A*lb.interpolate(sed[:,0], sed[:,1], magnitudes[j,i,0]/(redshift[j]+1), sort=False, out_of_bounds =np.nan)*magnitudes[j,i,0]/2.998e18
                  magnitudes[j,i,1]  = -2.5*np.log10(Fnu_gap) -48.6
                  magnitudes[j,i,2] = np.polyval(coefficients[i],magnitudes[j,i,1])

    return magnitudes
    
    
    
    
    
def find_nearest_filter(lum, filter_idx):
    save_index = np.argsort(abs(lum[:,0]-lum[filter_idx,0]))
    nearest =1    #just to avoid possible (?) infinte loop
    for idx in save_index:
        if lum[idx,1]==lum[idx,1]:   #filter with measured luminosity
           nearest = idx
           break
    return nearest

def host_correction_old(L, host_path ='~/DATA/Tables/galaxy_template.dat', control_negative = True, Niter=3):
    l5100 = lb.monochromatic_lum(L, 5100, out_of_bounds = 'extrapolate')
    l6156 = lb.monochromatic_lum(L, 6156, out_of_bounds = 'extrapolate')
    sed= pd.read_csv(host_path, header = None, sep =' ').to_numpy()
    sed_5100 = sed[:,1] / lb.interpolate(sed[:,0], sed[:,1], 5100, sort=False, out_of_bounds = 0)
    sed_6156 = sed[:,1] / lb.interpolate(sed[:,0], sed[:,1], 6156, sort=False, out_of_bounds = 0)
    deltaL = np.zeros(np.shape(L))
    N_bands =np.shape(L)[1]
    for j in range(0,np.shape(L)[0]):
    ##Richards+06 log(Lhost) = 0.87log(L_agn) + 2.887 L in erg/s Hz^-1
    # 4.7694 is to scale to nuFnu i.e. (1-0.87)*log(2.998e18/lambda) + 2.887
        if 0 < l5100[j] <10**44.75:
            agn =  l6156[j]
            for i in range(Niter):
                host=0.87*np.log10(agn)+4.7964          #vanden berk 2006 /richards 2006
                host =10**host
                agn = l6156[j]-host
            deltaL[j,:,1] = host*np.interp(L[j,:,0], sed[:,0], sed_6156, left=0, right=0)

                
        elif l5100[j]< 10**45.053:
            x = np.log10(l5100[j])-44
            ratio = 0.8052 -1.5502*x+0.9121*x*x-0.1577*(x**3)    #Shen et al. 2011
            host = (ratio*l5100[j])/(1+ratio)
            deltaL[j,:,1] = host*np.interp(L[j,:,0], sed[:,0], sed_5100, left=0, right=0)
        
    if control_negative:
        overestimated = deltaL[:,:,1] >= L[:,:,1]
        deltaL[np.any(overestimated, axis = 1), :, 1] = 0
        return deltaL, np.where(np.any(overestimated, axis = 1))[0]

    return deltaL


def host_correction(L, host_path ='~/DATA/Tables/galaxy_template.dat', control_negative = True, Niter=3):
    
    L5100 = lb.monochromatic_lum(L, 5100, out_of_bounds = 'extrapolate')
    L6156 = lb.monochromatic_lum(L, 6156, out_of_bounds = 'extrapolate')
    sed= pd.read_csv(host_path, header = None, sep =' ').to_numpy()
    
    sed[:,1] = sed[:,1]/np.interp(5100, sed[:,0], sed[:,1]) #sed normalized at 5100A°  
    host_f = interp1d(sed[:,0], sed[:,1], bounds_error=False ,fill_value=0)
    scale = 1/host_f(6156),
    deltaL = np.zeros(np.shape(L))
    
    host = get_host_luminosity(L5100, L6156, scale, Niter = Niter)
    
    for j in range(np.shape(L)[0]):
        
        deltaL[j,:,1] = host[j]*host_f(L[j,:,0])
        
    if control_negative:
        overestimated = deltaL[:,:,1] >= L[:,:,1]
        deltaL[np.any(overestimated, axis = 1), :, 1] = 0
        return deltaL, np.where(np.any(overestimated, axis = 1))[0]

    return deltaL


def get_host_luminosity(L5100, L6156, scale, Niter = 3):
    """Returns the Host luminosity at 5100 A.
       Scale = L5100/L6156
    """
    assert(len(L5100) == len(L6156))
    host_5100 = np.zeros((len(L5100,)))
    for j, (l5100, l6156) in enumerate(zip(L5100, L6156)):
    ##Richards+06 log(Lhost) = 0.87log(L_agn) + 2.887 L in erg/s Hz^-1
    # 4.7694 is to scale to nuFnu i.e. (1-0.87)*log(2.998e18/lambda) + 2.887
        if 0 < l5100 <10**44.75:
            agn =  l6156
            for i in range(Niter):
                host=0.87*np.log10(agn)+4.7964          #vanden berk 2006 /richards 2006
                host =10**host
                agn = l6156-host
            host_5100[j] = scale*host       # from 6156 to 5100

        elif l5100< 10**45.053:
            x = np.log10(l5100)-44
            ratio = 0.8052 -1.5502*x+0.9121*x*x-0.1577*(x**3)    #Shen et al. 2011
            host_5100[j] = (ratio*l5100)/(1+ratio)
    return host_5100  
  
    
    
def process_errors(magnitudes, minimum_error = 0.0, get_fit= True, deg = 3, shift_errors=False, missing_data_error =0.1):
    magnitudes[:,:,2] = np.maximum(magnitudes[:,:,2], minimum_error)  # set a minimum uncertainty value
    coefficients =[]
    if get_fit:
       mag=np.ma.MaskedArray(magnitudes[:,:,:], mask=np.isnan(magnitudes[:,:,:]))
       for i in range(magnitudes.shape[1]):
           coeff= np.ma.polyfit(mag[:,i,1], mag[:,i,2], deg) #interpolating errors on magnitudes to get similar values
           if shift_errors:
              variance= np.sqrt(np.nansum((mag[:,i,2]-np.polyval(coeff,mag[:,i,1]))**2)/mag.shape[0])
              for j in range(np.shape(magnitudes)[0]):
                  if np.polyval(coeff,magnitudes[j,i,1])-magnitudes[j,i,2]>=variance:      #shifting errors which deviate from the fit
                     magnitudes[j,i,2] = np.polyval(coeff,magnitudes[j,i,1])
           coefficients.append(coeff)
    else:
       for i in range(magnitudes.shape[1]):
           coeff = [0 for k in range(deg)]
           coeff.append(missing_data_error)
           coefficients.append(coeff)
    return magnitudes, coefficients
         















