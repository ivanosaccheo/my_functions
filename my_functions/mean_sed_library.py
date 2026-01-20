import os
import numpy as np 
from scipy import integrate
from scipy.interpolate import interp1d
from  my_functions import library as lb
from  my_functions import corrections 



PATH_TO_DATA = os.path.expanduser("~/WORK/my_functions")

def correct_magnitudes(redshift, filtro, emission_lines = True, IGM = True, DLA = True, 
                       spectrum_path = "tables/various/vanden_berk_13.dat"):
    """
    It returns an array with the magnitude corrections for the required filter.
    
    Redshift : iterable, with the redshifts of the sources
    filtro : object from library.filtro()
    emission_lines : bool, whether to apply corrections for the emission lines
    IGM : bool, whether to apply corrections for the InterGalctic Medium
    DLA : bool, whether to consider Deep Lyman Absorber in the IGM
    """
    filtro.get_transmission()
    lambda_obs = filtro.transmission[:,0]
    transmission = filtro.transmission[:,1]
    path = os.path.join(PATH_TO_DATA, spectrum_path)
    spectrum_rest = np.loadtxt(path, skiprows = 0) # 0 col= rest frame wav, 1 col = no Emission Lines, 2 col = with EL
    delta_M =[]
    for z in redshift:    
        delta_m_IGM =0
        delta_m_EL =0
        continuum, lines = shift_to_observed(spectrum_rest, z, lambda_obs)
        den = integrate.trapezoid(continuum*lambda_obs*transmission, lambda_obs)
        if IGM:
           tau = corrections.get_IGM_absorption(z, lambda_obs, DLA = DLA)
           y = np.exp(-tau)
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
   continuum = np.interp(lambda_obs, x, spectrum[:,1])
   lines = np.interp(lambda_obs, x, spectrum[:,2])

   return continuum, lines
        
        
def gap_filling(magnitudes, redshift,coefficients, SED_path = None):
    
    if SED_path is None:
        sed = lb.get_sed(which_sed = "krawczyk", which_type = "all")
        print("Using mean SED by Krawczyk+13 to perform gap repair")
    else:
        sed = np.loadtxt(SED_path, skiprows = 0) #lambda, lambdaL_lambda
    
    filled_magnitudes = np.copy(magnitudes)
    lack_data_all = np.isnan(filled_magnitudes[:,:,1])
    
    for i, (lack_data, z) in enumerate(zip(lack_data_all, redshift)):
        has_mag = filled_magnitudes[i, ~lack_data, :]   ##Just for easy reading
        lack_mag = filled_magnitudes[i, lack_data, :]   
        nearest_filter = [np.argmin(np.abs(wav - has_mag[:,0])) for wav in lack_mag[:,0]] ##find closest band with available data
        Fnu_has_mag = np.interp(has_mag[nearest_filter,0]/(z+1), sed[:,0], sed[:,1])*has_mag[nearest_filter,0]
        Fnu_lack_mag = np.interp(lack_mag[:,0]/(z+1), sed[:,0], sed[:,1])*lack_mag[:,0]
        filled_magnitudes[i, lack_data, 1] = has_mag[nearest_filter,1] -2.5*np.log10(Fnu_lack_mag/Fnu_has_mag)
    
    for j, coefficient in enumerate(coefficients):     #iterating over the bands, adding uncertainties
        filled_magnitudes[lack_data_all[:,j], j, 2] = np.polyval(coefficient, filled_magnitudes[lack_data_all[:,j], j, 1])

    return filled_magnitudes
   
def host_correction(L, control_negative = True, Niter=3):
    
    L5100 = lb.monochromatic_lum(L, 5100, out_of_bounds = 'extrapolate')
    L6156 = lb.monochromatic_lum(L, 6156, out_of_bounds = 'extrapolate')
    sed= lb.get_host()
    
    sed[:,1] = sed[:,1]/np.interp(5100, sed[:,0], sed[:,1]) #sed normalized at 5100A°  
    host_f = interp1d(sed[:,0], sed[:,1], bounds_error=False ,fill_value=0)
    scale = 1/host_f(6156)
    deltaL = np.zeros(np.shape(L))
    
    host = corrections.get_host_luminosity(L5100, L6156, scale, Niter = Niter)
    
    for j in range(np.shape(L)[0]):
        
        deltaL[j,:,1] = host[j]*host_f(L[j,:,0])
        
    if control_negative:
        overestimated = deltaL[:,:,1] >= L[:,:,1]
        deltaL[np.any(overestimated, axis = 1), :, 1] = 0
        return deltaL, np.any(overestimated, axis = 1)

    return deltaL


def process_errors(magnitudes, minimum_error = 0.0, get_fit= True, deg = 3, shift_errors=False, missing_data_error = 0.1):
    pro_magnitudes = np.copy(magnitudes)
    pro_magnitudes[:,:,2] = np.maximum(pro_magnitudes[:,:,2], minimum_error)  # set a minimum uncertainty value
    coefficients =[]
    if get_fit:
       mag=np.ma.MaskedArray(pro_magnitudes[:,:,:], mask=np.isnan(pro_magnitudes[:,:,:]))
       for j in range(pro_magnitudes.shape[1]):
           coeff= np.ma.polyfit(mag[:,j,1], mag[:,j,2], deg) #interpolating errors on magnitudes to get similar values
           if shift_errors:
              variance= np.sqrt(np.nansum((mag[:,j,2]-np.polyval(coeff,mag[:,j,1]))**2)/mag.shape[0])
              for i in range(np.shape(pro_magnitudes)[0]):
                  if np.polyval(coeff, pro_magnitudes[i,j,1])- pro_magnitudes[i,j,2]>=variance:      #shifting errors which deviate from the fit
                     pro_magnitudes[i,j,2] = np.polyval(coeff,pro_magnitudes[i,j,1])
           coefficients.append(coeff)
    else:      ##  Constant error (i.e. polynomial of 0 degree)
       for j in range(pro_magnitudes.shape[1]):
           coeff = [0 for k in range(deg)]
           coeff.append(missing_data_error)
           coefficients.append(coeff)
    
    return pro_magnitudes, coefficients