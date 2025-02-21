##### Utili 
# interpolate    equal to np.interp :(
# two_2_three    transform tables into 3d array
# three_2_two    inverse of two_2_three
# compute_mean_in_bins   compute the mean and variance in the provided bins

### Astro
# filtro             class to read filters
# get_luminosity     from AB magnitudes to luminosity (or fluxes)
# get_magnitudes     from luminosities to AB magnitudes 
# monochromatic_lum  compute monochromatic luminosities
# merge_bands        merge photometry in the same band (e.g. ukidss K and 2mass K) into one column !!!!!to_be_improved!!!!
#L_2_abs_mag / abs_mag2_L conversion between lumnosities and absolute magnitudes

### AGN/sed related
# get_sed            get AGN sed template
# get_host           get host_template
#find_normalization  Single component sed fitting
#lusso_recipe        compute the sed between 911/1216 A° and 1kev using the same recipe as in Lusso+10
#get_xray_luminosity compute the xray luminosity using the alpha_OX by Lusso+10/16
#move_xray_band         compute the Xray luminosity from another xray band
#get_integrated_xray    compute the integrated xray luminosity







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy import units
from astropy.cosmology import FlatLambdaCDM
from astropy import constants
import datetime 
import pytz



PATH_TO_DATA =  "my_functions"



def interpolate(x, y, x0, out_of_bounds='error', sort=True, log_log=True):
    """
    It returns the value of y computed at x0 linearly interpolating between 
    two adjacent points. x and y must have the same size.
    x :  N*1 array
    y :  N*1 array
    x0 : float
    out_of_bounds : Number, np.nan, 'extrapolate', 'error'. It determines the behaviour of the interpolation when x0 is out of bounds

    sort: logical, if true the templates are sorted before computation

    log_log : logical, if true it interpolates in the log-log space

    """
    if not isinstance(x, np.ndarray): x = np.array(x)
    if not isinstance(y, np.ndarray): y = np.array(y)  
    
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    if sort:  # sorting the templates
       order = np.argsort(x)
       x, y = x[order], y[order]
    
    N = len(x)-1
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
                     other_features = data[col_to_strip]
                     raw_features = [i for i in data.columns if i not in col_to_strip]
                     raw_data = data[raw_features].to_numpy()
    
            elif all(isinstance(i, str) for i in extra_features):
                for name in extra_features: assert(name in data.columns)
                other_features = data[extra_features]
                raw_features = [i for i in data.columns if i not in extra_features]
                raw_data = data[raw_features].to_numpy()
            else: 
                raise Exception('La lista di feature deve essere composta o solo da stringhe o da interi')
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
        raise Exception('Data ust be a Pandas DataFrame or a numpy 2D array')
     
    Nqso = raw_data.shape[0]
    if has_wavelength:  
       Nbands, Nproperties = int(raw_data.shape[1]/3), 3
    else:
       Nbands, Nproperties = int(raw_data.shape[1]/2), 2
    
    NewData =np.zeros((Nqso,Nbands, Nproperties))

    for i in range(Nbands):
            for k in range(Nproperties):
                NewData[:,i,k] =raw_data[:,i*Nproperties+k]      
               
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
    Nqso = data.shape[0]
    Nbands = data.shape[1]
    Ndata = data.shape[2]
    
    NewData = np.zeros((Nqso, Nbands*Ndata+len(args)))
    
    for i in range(Nbands):
        for k in range(Ndata):
            NewData[:, i*Ndata+k] =data[:,i,k]
    if args: 
        for i, feature in enumerate(args):
            NewData[:, Ndata*Nbands+i] = feature
    
    name_array = None
    if all_names:
        name_array = all_names
    
    
    elif band_names and Ndata == 3:
       name_array =[]
       for i, name in enumerate(band_names):
           if i <  Nbands:
               name_array.append('lambda_'+name)
               name_array.append(name)
               name_array.append('err_'+name)
           else:
               name_array.append(name)

    elif band_names and Ndata ==2:
       name_array =[]
       for i, name in enumerate(band_names):
           if i <  Nbands:
               name_array.append(name)
               name_array.append('err_'+name)
           else:
               name_array.append(name)
        
    return pd.DataFrame(NewData, columns = name_array)  

def compute_mean_in_bins(x, y, bins, function = 'mean'):
    
    digitized= np.digitize(x, bins)
    if function == 'mean':
        x_mean = np.asarray([np.nanmean(x[digitized == i]) for i in range(1, len(bins))])
        x_var =  np.asarray([np.nanvar(x[digitized == i]) for i in range(1, len(bins))])
        y_mean = np.asarray([np.nanmean(y[digitized == i]) for i in range(1, len(bins))])
        y_var = np.asarray([np.nanvar(y[digitized == i]) for i in range(1, len(bins))])
    elif function == 'median':
        x_mean = np.asarray([np.nanmedian(x[digitized == i]) for i in range(1, len(bins))])
        x_var =  np.asarray([np.nanvar(x[digitized == i]) for i in range(1, len(bins))])
        y_mean = np.asarray([np.nanmedian(y[digitized == i]) for i in range(1, len(bins))])
        y_var = np.asarray([np.nanvar(y[digitized == i]) for i in range(1, len(bins))])
        print('Sto ancora calcolando la varianza, non la MAD')
    else:
        print('function must be mean or median')
        return None


    N = np.asarray([np.sum(digitized == i) for i in range(1, len(bins))])
    return np.stack([x_mean, y_mean, x_var, y_var, N], axis =1)

############# Astro

def get_flux(magnitudes):

    fluxes = np.zeros(magnitudes.shape)
    
    if len(fluxes.shape) == 2:
        fluxes[:, 1] = 10**(-0.4*(magnitudes[:,1] +48.6))*(2.998e18/magnitudes[:,0]) #magntitudes to fluxes
        fluxes[:, 2] = fluxes[:,1] * magnitudes[:, 2]*0.4*np.log(10)   #error on fluxes
        fluxes[:, 0] = magnitudes[:,0]       #observed frame wavelengths
    
    elif len(magnitudes.shape) == 3:    
        for i in range(magnitudes.shape[1]):
            fluxes[:, i, 1] = 10**(-0.4*(magnitudes[:,i,1] + 48.6))*(2.998e18/magnitudes[:,i,0]) #magntitudes to fluxes
            fluxes[:, i, 2] = fluxes[:,i,1] * magnitudes[:, i, 2]*0.4*np.log(10)   #error on fluxes
            fluxes[:, i, 0] = magnitudes[:,i,0]      #observed frame wavelengths
    else:
        raise Exception('wrong format for magnitudes')
    
    return fluxes       


def get_luminosity(magnitudes, redshift, H0 = 70, Om0 = 0.3):         

    luminosity = get_flux(magnitudes)
    
    if len(magnitudes.shape) == 2:
        dl = FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance(redshift).to(units.cm).value 
        luminosity[:,0] = luminosity[:,0]/(redshift+1) #rest frame wavelengths
        luminosity[:, 1:] = luminosity[i, 1:]*(dl*dl*4*np.pi).reshape(-1,1) 

    else: 
        dl = FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance(redshift).to(units.cm).value    
        for i in range(magnitudes.shape[1]):
            luminosity[:, i, 0] = luminosity[:,i,0]/(redshift+1)       #rest frame wavelengths
            luminosity[:, i, 1:] = luminosity[:, i, 1:]*(dl*dl*4*np.pi).reshape(-1,1)
    return luminosity         


def get_magnitudes(luminosity, redshift, H0 =70, Om0 = 0.3):
    magnitudes = np.zeros(luminosity.shape)
    
    if len(magnitudes.shape) == 2:
        dl = FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance(redshift).cgs.value 
        
        magnitudes[:,0] = luminosity[:,0]*(redshift+1)
        magnitudes[:,1:] = luminosity[:,1:]/(dl*dl*4*np.pi)
       
        magnitudes[:,1:] = magnitudes[:,1:]*magnitudes[:,0].reshape(-1,1)/2.998e18  #Fnu

        magnitudes[:,2] = 2.5*(magnitudes[:,2]/magnitudes[:,1])/np.log(10)
        magnitudes[:,1] = -2.5*np.log10( magnitudes[:,1]) -48.6

     
    if len(magnitudes.shape) == 3:    
        dl = FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance(redshift).cgs.value 
        for i in range(magnitudes.shape[1]):
            magnitudes[:, i, 0] = luminosity[:, i, 0]*(redshift+1)
            magnitudes[:, i, 1:] = luminosity[:, i, 1:]/(dl*dl*4*np.pi).reshape(-1,1)
        
            magnitudes[:, i, 1:] = magnitudes[:, i, 1:]*magnitudes[:, i, 0].reshape(-1,1)/2.998e18  #Fnu
             
            magnitudes[:, i, 2] = 2.5*(magnitudes[:, i, 2]/magnitudes[:, i, 1])/np.log(10)
            magnitudes[:, i, 1] = -2.5*np.log10( magnitudes[:, i, 1]) -48.6
     
    return magnitudes


def monochromatic_lum(data, wavelength, uncertainties = False, out_of_bounds = np.nan):
    
    if len(data.shape) == 2: # single object
        lum= interpolate(data[:,0],data[:,1], wavelength, out_of_bounds=out_of_bounds)
        if uncertainties: 
            lum_low = interpolate(data[:,0],data[:,1] - data[:,2], wavelength, out_of_bounds=out_of_bounds)
            lum_up = interpolate(data[:,0], data[:,1] + data[:,2], wavelength, out_of_bounds=out_of_bounds)
            lum = np.hstack(lum, lum_low, lum_up)
    
    elif len(data.shape) == 3:
        lum = np.array([interpolate(d[:,0],d[:,1], wavelength, out_of_bounds=out_of_bounds) for d in data])
        if uncertainties: 
            lum_low = np.array([interpolate(d[:,0],d[:,1]-d[:,2], wavelength, out_of_bounds=out_of_bounds) for d in data])
            lum_up = np.array([interpolate(d[:,0],d[:,1]+d[:,2], wavelength, out_of_bounds=out_of_bounds) for d in data])
            lum = np.vstack([lum, lum_low, lum_up]).T
    return lum  

def add_wavelength(magnitudes, wavelen):
    assert(len(wavelen)== magnitudes.shape[1])
    new_magnitudes = np.zeros((magnitudes.shape[0],magnitudes.shape[1], 3))
    new_magnitudes[:,:,0] = wavelen
    new_magnitudes[:,:,1:]=magnitudes
    return new_magnitudes


def merge_bands(df, column_name):
    if isinstance(column_name, str):
        new_column = df[column_name].to_numpy()
    else:
        new_column = df[column_name[0]].to_numpy()
        for col in column_name[1:]:
            where_nan = np.isnan(new_column)
            new_column[where_nan] = df[col][where_nan].to_numpy()
    return new_column


class filtro():
    
    def __init__(self, filter_name, path = 'tables/filters'):
        self.path = os.path.join(PATH_TO_DATA,path)
        self.get_filter_name(filter_name)
        if hasattr(self, 'name'):
            self.wav = self.get_effective_wavelength()
        
        return None 

        
    def get_filter_name(self, filter_name):
        names = [i for i in os.listdir(self.path) if i.endswith('.dat')]
        matching_names = [i for i in names if filter_name.casefold() in i.casefold()]
        
        if len(matching_names) == 1:
            self.name = matching_names[0][:-4]
            self.filename = matching_names[0]
            return None
        elif len(matching_names) > 1:
            matching_names.sort()
            print(f"Multiple filters with {filter_name} name:")
            for name in matching_names: print(name)
            return None
        
        elif len(matching_names) == 0:
            print(f"No filter with {filter_name} name")   
            return None
      
    def get_effective_wavelength(self):
        table = pd.read_csv(os.path.join(self.path, "filter_list.txt"), sep = "\s+")
        eff_wav = float(table[table['Name'] == self.name]['eff_wavelength'].iloc[0])
        return eff_wav
        
    def get_transmission(self):
        self.transmission = np.loadtxt(os.path.join(self.path, self.filename))
        self.wav_min = np.min(self.transmission[self.transmission[:,1]>0,0])
        self.wav_max = np.max(self.transmission[self.transmission[:,1]>0,0])
    
    def convolve(self, wavelengths, f_lambda, return_magnitude = True,
              left = 0, right = 0):
        """
        Output : magnitude if return_magnitude = False, else lambda * F_lambda at the effective wavelength
        of the filter.
        No zero point so it must be used just for colors (???)
        f_lambda = flux in erg/s cm^-2 A°^-1
        wavelengths = wavelength of f_lambda
        left, right = per np.interp se il flusso non compre tutto l'intervallo della trasmissione del filtro
        """
        if not hasattr(self, "transmission"):
            self.get_transmission()

        f_lambda_filter = np.interp(self.transmission[:,0], wavelengths, f_lambda, 
                                    left = left, right = right)
        numeratore = np.trapz(f_lambda_filter*self.transmission[:,1]*self.transmission[:,0], 
                              self.transmission[:,0])/2.998e18
        denominatore = np.trapz(self.transmission[:,1]/self.transmission[:,0], self.transmission[:,0])
        f_nu = numeratore/denominatore
        if return_magnitude:
            return -2.5 * np.log10(f_nu) - 48.6
        else:
            return (f_nu/self.wav)*2.998e18

def abs_mag_2_L(abs_M, wavlen):
    nuFnu = (10**(-0.4*(abs_M+48.6)))*2.998e18/wavlen
    d = 10*constants.pc.cgs.value
    return nuFnu*4*np.pi*d*d

def L_2_abs_mag(L, wavlen):
    d = 10*constants.pc.cgs.value
    fnu = (L/(4*np.pi*d*d))*wavlen/2.998e18
    return -2.5*np.log10(fnu) -48.6

            
 ########### AGN /SED

def get_sed(which_sed='krawczyk', which_type='All', normalization=False, log_log=False, path= 'tables/sed_templates'):
   
    path = os.path.join(PATH_TO_DATA ,path)
   
    if 'krawczyk' in which_sed.lower():
        sed = pd.read_csv(os.path.join(path,'krawczyk_13.dat') , sep=' ', header=0, comment ='#')
        sed_types = [i for  i in sed.columns[1:] if "sigma" not in i]
        if which_type.casefold()  not in sed_types:
            raise Exception(f"which_type must be one of {sed_types}")
        x, y = sed["lambda"].to_numpy(), sed[which_type.casefold()].to_numpy()
    
    elif 'wissh' in which_sed.lower():
        sed = pd.read_csv(os.path.join(path,'wissh_S23.dat') , sep=' ', header=0)
        x, y  = sed['lambda'].to_numpy(), sed["L"].to_numpy()
    
    elif 'richards'in which_sed.lower():
        SED = pd.read_csv(os.path.join(path,'richards_06.dat') , sep=' ', header=0, comment ='#')
        sed_types = [i for  i in sed.columns[1:] if "sigma" not in i]
        if which_type.casefold()  not in sed_types:
            raise Exception(f"which_type must be one of {sed_types}")
        x, y = sed["lambda"].to_numpy(), sed[which_type.casefold()].to_numpy()
            
    elif "polletta" in which_sed.lower():
        path = os.path.join(path, "polletta")
        if "all" in which_type.lower():
            available_sed = [i for i in os.listdir(path) if i.endswith(".sed")]
            print("Available SEDs from Polletta are:")
            for name in available_sed: print(name.replace("_template_norm.sed", ""))
            return None
        else:
            fname = os.path.join(path,f"{which_type}_template_norm.sed")
            try:
                sed = pd.read_csv(fname, header = None, sep='\s+').to_numpy()
                x, y = sed[:,0], sed[:,1]*sed[:,0] # lambda*F_lambda
            except FileNotFoundError:
                print(f"{which_type} not found, available SEDs from Polletta are:")
                available_sed = [i for i in os.listdir(path) if i.endswith(".sed")]
                for name in available_sed:
                     print(name.replace("_template_norm.sed", ""))
                raise Exception 
    
    elif "berk" in which_sed.lower():
        sed = pd.read_csv(os.path.join(path,'vandenberk_01.dat') , sep='\s+', header=0)
        x = sed["lambda"].to_numpy(), 
        y = x*sed["f_lambda"].to_numpy()    

    elif "caballero" in which_sed.lower():
        sed = pd.read_csv(os.path.join(path,'hernan_caballero_17.dat') , sep=' ', header=0, comment = '#')
        sed_types = [i for  i in sed.columns[1:] if "sigma" not in i]
        if which_type.casefold()  not in sed_types:
            raise Exception(f"which_type must be one of {sed_types}")
        x, y = sed["lambda"].to_numpy(), sed[which_type.casefold()].to_numpy()
    else:
        raise Exception("Which_sed can be 'wissh', 'krawczyk', 'richards' 'polletta', 'vandenberk', 'caballero'")

    if normalization:
        norm = normalization[1]/np.interp(normalization[0], x, y)
        y = y*norm
    if log_log:
        x, y = np.log10(x), np.log10(y)

    sed = np.vstack([x,y]).T
    return sed


def get_host(path ='tables/sed_templates/host_galaxy_sed.csv'):
     path = os.path.join(PATH_TO_DATA, path)
   
     return pd.read_csv(path, header = 0, sep = ',' ).to_numpy()


def find_normalization(wavelengths, L, err_L, sed, lambda_min=1216, lambda_max=50000):
    """
    fitta i punti di un quasar assumendo che siano modellabili attraverso una sigola SED opportunamente normalizzata 
    Parameters
    ----------
    wavelengths : array/list rest-frame wavelengths of the observed QSO 
    
    L :  array/list lambda*L of the observed QSO
    
    err_L: array/list error on L
    
    sed : numpy N*2 array
          array con la sed da fittare sed[:,0] = lambda, sed[:,1] = lambda*L 
    lambda_min : float, optional
                 Nel fit vengono considerati solo i punti con lambda>lambda_min.The default is 1216.
    lambda_max : float, optional
                  Nel fit vengono considerati solo i punti con lambda<=lambda_max. The default is 50000.

    Returns
    -------
    norm : float
           Costante normalizzazione best fit
    norm_min : float
               costante di normalizzazione minima tale che il fit stia a un Delta chi^2 <= 1 rispetto al best
    norm_max : float
               costante di normalizzazione massima tale che il fit stia a un Delta chi^2 <= 1 rispetto al best
    chi2 : Float
        Reduced Chi square of the best fit

    """
    
    x = np.asarray(wavelengths)
    y = np.asarray(L)
    dy = np.asarray(err_L)
    
    inbounds = np.logical_and(x>= lambda_min, x <= lambda_max)
    x, y, dy = x[inbounds], y[inbounds], dy[inbounds]
    x = x[~np.isnan(y)]
    dy = dy[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    dof = len(y)-1
    template = []
    
    for wav in x:
        template.append(interpolate(
            sed[:, 0], sed[:, 1], wav, out_of_bounds='0'))
    template = np.asarray(template)
    Syf = np.sum((y*template)/(dy*dy))
    Sff = np.sum((template*template)/(dy*dy))
    Syy = np.sum((y*y)/(dy*dy))
    norm = Syf/Sff
    chi2 = np.sum((y-norm*template)**2/(dy*dy))/dof
    Delta = (Syf**2)-(Sff*(Syy-dof*(chi2+1)))
    norm_min = (Syf - np.sqrt(Delta))/Sff
    norm_max = (Syf+np.sqrt(Delta))/Sff
    return norm, norm_min, norm_max, chi2

def lusso_recipe(lambda_start, L_start, L_1kev, Npoints = 30):
    """
     fornisce la sed tra lambda_start e lambda = 1 keV come Lusso+10:
        -lambda L ~ lambda^0.8 tra lambda_start e 500 A°
        -retta con slope variabile tra 500 A° e 1 keV

    """
    assert(lambda_start >500)
    x = np.logspace(np.log10(lambda_start), np.log10(500), int(Npoints/2))
    A = L_start/(lambda_start**0.8)
    y = (A*(x**0.8))
    
    sed_1 = np.stack([x,y], axis =1)
    
    x0, y0 = np.log10(500), np.log10(A*500**0.8)
    x1, y1 = np.log10(12.398), np.log10(L_1kev)
    
    x = np.logspace(x0, x1, (Npoints-int(Npoints/2)))
    A = ((y1-y0)/(x1-x0))
    y = y0 + A*(np.log10(x)-x0)
    y = 10**y
    
    sed_2 = np.stack([x,y], axis =1)
    
    sed =np.concatenate([sed_1, sed_2], axis = 0)
    sed = sed[sed[:,0].argsort()]
    return sed

def move_xray_band(Lstart, energy_start, energy_final, photon_index = 1.8):
    """
    Computes the Xray luminosity from one band (energy_start) to another (energy_final)
    Lstart = luminosity in erg/s
    energY_start/energy_final = wavlengths in keV
    """
    wav_start = 12.398/energy_start
    wav_final = 12.398/energy_final
    return Lstart*(wav_final/wav_start)**(photon_index-2)

def get_xray_luminosity(L2500, energy = 2, photon_index = 1.8, 
                            recipe = "lusso+16"):
    """
    Deriva la luminosita X a energia = energy a partire dalla relazione L_UV-L_x
    Lx è trasformata da 2 kev all'energia specificata da energy assumendo un photon index Gamma
    lambda*L = lambda^(Gamma-2)

    Parameters
    ----------
    l2500 : Float
            2500 A° luminosity in erg/s
    energy : Float, optional
             banda in kev a cui calcolare Lx. The default is 2.
    photon_index : float, optional
             Photon index 
    recipe : string, optional
             which parameters to use to derive Lx, either lusso+10 or lusso+16

    Returns
    Lx
    """
    parametri = {"lusso+16" : [0.642, +6.965],
                 "lusso+10" : [0.599, +8.275]}
    if recipe not in parametri.keys():
        raise Exception(f"recipe must be among {[i for i in parametri.keys()]}")
    alpha, beta = parametri[recipe]
    l2500 = np.log10(L2500*(2500/2.998e18))
    l2kev =alpha*l2500+beta
    l2kev = ((10**l2kev)*(2.998e18/6.199))
    return move_xray_band(l2kev, 2, energy, photon_index = photon_index)


def get_integrated_xray(L_start, energy_start, energy_1 = 2, energy_2 = 10, photon_index = 1.8):
    """
    Calcola la luminnosità intrgrata tra energy_1 e energy_2 a partire da una luminosità L_start a
    energy_start. Assume che L_lambda \propoto \lambda^(photon_index-3)
    """
    ### L_lambda ~ lambda^gamma-3
    if photon_index != 2:
        L_1 = move_xray_band(Lstart= L_start, energy_start=energy_start, energy_final=energy_1, 
                             photon_index = photon_index)
        L_2 = move_xray_band(Lstart= L_start, energy_start=energy_start, energy_final=energy_2, 
                             photon_index = photon_index)
        return np.abs((1/(photon_index-2))*(L_1-L_2))   ##abs in case energies are not sorted
    else:
        return np.abs(L_start*np.log(energy_1/energy_2))

def get_mono_xray_from_integrated(energy, L_integrated, energy_1 = 2, energy_2 = 10,
                                  photon_index = 1.8):
    """calcola la luminoxita monocromatica in lambda*L_lambda alle energie 'energy'
       a partire da una luminosità integrata tra energy_1 e energy_2
    """
    wav_start = 12.398/energy_1
    wav_final = 12.398/energy_2
    wav = 12.398/np.array(energy)
    if photon_index != 2:
        normalization = (photon_index-2)*L_integrated
        normalization = normalization/(wav_start**(photon_index-2)-wav_final**(photon_index-2))
    else:
        normalization = L_integrated/np.log(wav_start/wav_final)

    normalization = np.abs(normalization)
    L = normalization*(wav**(photon_index-2))
    return L
    
class quasar_lines:
    """Loads Table 2 (list of all observed lines in QSO spectrum) in Vanden Berk+2001"""
    def __init__(self, maxrows = 20, flux_sorted = True, remove_iron = False,
                 wavmin = None, dropped_columns = None):
        path = os.path.join(PATH_TO_DATA,"tables/various","vanden_berk_2001_tab2.dat")
        self.table = pd.read_csv(path, sep=' ', comment="#")

        if wavmin is not None:
            self.table = self.table[self.table["obs_wav"]>= wavmin]
        if flux_sorted:
            self.table.sort_values(by="flux", inplace = True, ascending = False)
        if remove_iron:
            self.table = self.table[~self.table["ID"].str.contains("Fe")]
        if maxrows is not None and maxrows<= len(self.table):
            self.table = self.table.iloc[:maxrows, :]
        if dropped_columns is not None:
            self.table = self.table.drop(columns = dropped_columns)
        self.table = self.table.reset_index(drop = True)
        
        return None
    
    def get_plot_ID(self):
        new_names = [name.replace("{" ,"$\\") for name in self.table["ID"]]
        new_names = [name.replace("}" ,"$") for name in new_names]
        self.table["plot_ID"] = new_names
        return None
    
def get_quasar_lines(maxrows = 25, flux_sorted = True, remove_iron = True,
                     wavmin = None,
                 dropped_columns = ["u_ID",	"f_ID", "e_obs_wave", "e_flux", "e_flux",
                                    "f_width", "skew", "e_EW"]):
    qso_lines = quasar_lines(maxrows = maxrows, flux_sorted=flux_sorted, wavmin = wavmin,
                             remove_iron = remove_iron, dropped_columns = dropped_columns)
    qso_lines.get_plot_ID()
    return qso_lines.table






   
    
   
  




















  







































  



















