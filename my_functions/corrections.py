import os
import numpy as np
from scipy import integrate
import numpy.polynomial.polynomial as poly
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
from  my_functions import library as lb


PATH_TO_DATA = lb.PATH_TO_DATA
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



def get_host_luminosity(L5100, L6156, scale, Niter = 3):
    """Returns the Host luminosity at 5100 A using the Richards+06 and Shen+11 relations
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
    norm = equivalent_width/np.trapezoid(line/continuum, xx)
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











