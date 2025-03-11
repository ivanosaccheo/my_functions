import os
import numpy as np
from scipy import integrate
import numpy.polynomial.polynomial as poly
from  my_functions import library as lb
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats


PATH_TO_DATA = os.path.expanduser("~/WORK/my_library")
#os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    return tau


def lyman_continuum_DLA(redshift, lambda_obs):
    
    ll = 911.8                   #lyman-limit
    wav = lambda_obs/ll
    tau = np.zeros((len(lambda_obs),))
    
    if redshift < 2:
        idx = wav < (1+redshift) 
        tau[idx] = 0.211*((1+redshift)**2) - 0.0766*((1+redshift)**2.3)*(wav[idx]**(-0.3))-0.135*(wav[idx]**2) 

    else:
        idx1 = wav < 3
        idx2 = np.logical_and(wav >=3, wav < (1+redshift))
   
        tau[idx1] = (0.634 + 0.047*((1+redshift)**3) -0.0178*((1+redshift)**3.3)*(wav[idx1]**(-0.3))
                    -0.135*(wav[idx1]**2)-0.291*(wav[idx1]**(-0.3)))
           
        tau[idx2] = 0.047*((1+redshift)**3)-0.0178*((1+redshift)**3.3)*(wav[idx2]**(-0.3))-0.0292*(wav[idx2]**3)
    
    return tau
    

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

def get_lyman_coefficients(coefficients_path= "tables/various/lyman_series_coefficients.dat"):
    path = os.path.join(PATH_TO_DATA,coefficients_path)
    return np.loadtxt(path)
    
    
def get_IGM_absorption(redshift, lambda_obs, coefficients = get_lyman_coefficients(), DLA = True):
    """ 
    Optical depth computed according to Inoue et al. 2014
    """
    tau_continuum_laf = lyman_continuum_LAF(redshift, lambda_obs)
    
    tau_series_laf = lyman_series_LAF(redshift, lambda_obs, coefficients)
    if not DLA:
         return  tau_continuum_laf + tau_series_laf 
    
    else:  
        tau_continuum_dla = lyman_continuum_DLA(redshift, lambda_obs)
        tau_series_dla = lyman_series_DLA(redshift, lambda_obs, coefficients)
        return  tau_continuum_laf +  tau_continuum_dla + tau_series_laf + tau_series_dla


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
           tau = get_IGM_absorption(z, lambda_obs, DLA = DLA)
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
    
    host = get_host_luminosity(L5100, L6156, scale, Niter = Niter)
    
    for j in range(np.shape(L)[0]):
        
        deltaL[j,:,1] = host[j]*host_f(L[j,:,0])
        
    if control_negative:
        overestimated = deltaL[:,:,1] >= L[:,:,1]
        deltaL[np.any(overestimated, axis = 1), :, 1] = 0
        return deltaL, np.any(overestimated, axis = 1)

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
         

###### Reddening laws
def calzetti_2000(wavlen, Rv = 4.05):
    k_lambda = np.ones(len(wavlen))
    logic = (wavlen <= 6300)
    l1 =  wavlen[logic]/1e4
    l2 = wavlen[~logic]/1e4
    k_lambda[logic] =  2.659*(-2.156 + 1.509/l1 - 0.198/(l1*l1) +0.011/(l1*l1*l1)) + Rv
    k_lambda[~logic] = 2.659*(-1.857 +1.040/l2) + Rv
    return k_lambda

def prevot_1984(wavlen):
    return 1.39 * ((wavlen/1e4)**(-1.2))

def charlot_2000(wavlen, ism_fraction = 0.6):
    ism_fraction = np.clip(ism_fraction, 0, 1)
    slope_ism = -0.7
    slope_bc = -1.3
    k_lambda_ism = ism_fraction*(wavlen**slope_ism)
    k_lambda_bc = (1-ism_fraction)*(wavlen**slope_bc)
    k_lambda =  k_lambda_ism+k_lambda_bc
    wavlen_v = 5431.91   ## Vimos 
    wavlen_b = 4288.94   ## Vimos
    f = interp1d(wavlen, k_lambda, fill_value='extrapolate')
    norma = 1/(f(wavlen_b)- f(wavlen_v)) #k(B)-k(V) = 1
    
    return norma*k_lambda

class reddening_law:

    def __init__(self, ebv = 0, Av = None, law = "calzetti", Rv = "default",
                 ism_fraction = 0.5):
        self.ebv = ebv
        self.Av = Av
        self.law = law.casefold()
        if Rv == "default":
            Rv_dict = {"calzetti" : 4.05, "prevot" : 2.72, "charlot" : 3.1}
            self.Rv = Rv_dict[self.law]
        else: 
            self.Rv = Rv
        if self.Av is None: self.update_Av()
        else: self.update_ebv()
        self.ism_fraction = ism_fraction
        return None
    
    def get_k_lambda(self, wavlen):
        if "calzetti" in self.law:
            self.k_lambda = calzetti_2000(wavlen, Rv = self.Rv)
        elif "prevot" in self.law:
            self.k_lambda = prevot_1984(wavlen)
        elif "charlot" in self.law:
            self.k_lambda = charlot_2000(wavlen, ism_fraction=self.ism_fraction)
        else: 
            raise Exception("law must be 'calzetti', 'prevot' or 'charlot'")
        return None
    
    def get_tau_lambda(self, wavlen, control_negative = True):
        if not hasattr(self, "k_lambda"):
            self.get_k_lambda(wavlen)
        self.tau_lambda = (self.k_lambda*self.ebv)/1.086
        if control_negative:
             self.tau_lambda[self.tau_lambda<0] =0
        return None 
    
    def get_A_lambda(self, wavlen, control_negative = True):
        if not hasattr(self, "k_lambda"):
            self.get_k_lambda(wavlen)
        self.A_lambda = (self.k_lambda*self.ebv)
        if control_negative:
             self.A_lambda[self.A_lambda<0] =0
        return None
        
    def get_extinction(self, wavlen):
        if not hasattr(self, "k_lambda"):
            self.get_k_lambda(wavlen)
        self.get_tau_lambda(wavlen)
        self.extinction = np.exp(-self.tau_lambda)
        return None
    
    def update_ebv(self):
        self.ebv = self.Av/self.Rv
        return None
    
    def update_Av(self):
        self.Av = self.ebv*self.Rv
        return None


def get_line_normalization_vandenberk(obs_wav, wav_min, wav_max, equivalent_width, stddev):
    ### Only used in get_lines_vandenberk
    xx = np.linspace(wav_min, wav_max, 1000)
    continuum = get_continuum_vandenberk(xx)
    line = stats.norm.pdf(xx, loc = obs_wav, scale = stddev)
    norm = equivalent_width/np.trapz(line/continuum, xx)
    return np.abs(norm)
    

def get_continuum_vandenberk(xx, wav_break = 5300):
    alpha_lambda_1 = -1.56
    alpha_lambda_2 = 0.45
    K = wav_break**(alpha_lambda_1-alpha_lambda_2)
    continuum = np.where(xx<=wav_break, xx**alpha_lambda_1, K*xx**alpha_lambda_2)
    return continuum

def get_lines_vandenberk(wavlen, table):
    xx = np.repeat(np.expand_dims(wavlen, axis =1), len(table), axis =1)
    lyalpha_normalization = get_line_normalization_vandenberk(1216.25, 1160, 1290, 92.91, 19.46)
    lines_luminosity = (table["flux"]*lyalpha_normalization/100).to_numpy()
    template = (lines_luminosity*stats.norm.pdf(xx, loc = table["obs_wav"].to_numpy(), scale=table["width"].to_numpy())).sum(axis =1)
    return template











