import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy import units
from astropy.cosmology import FlatLambdaCDM
import datetime 
import pytz

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
    

def compute_xray_luminosity(l2500, energy=2, photon_index=1.7):
    """
    Deriva la luminosita X a partire dalla relazione di Lusso+10 Log(Lx)   = 0.599 Log(Luv) +8.275.
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

    Returns
    Lx
    """
    wav = 12.398/energy
    # Trasformo in L_nu e prendo il log
    l2500 = np.log10(l2500*(2500/2.998e18))
    l2kev = 0.599*l2500+8.275
    l2kev = ((10**l2kev)*(2.998e18/6.199))  # trasformo in lambda*L
    A = l2kev/(6.199**(photon_index-2))
    lx = (wav**(photon_index-2))*A
    return lx


def get_sed(which_sed='WISSH', which_type='All', normalization=False, log_log=False, add_path=False, path= 'Tavole_utili/'):
    if 'krawczyk' in which_sed.lower():
        path = path+'krawczyk_13_all.csv'
        if add_path:
           path = add_path+path
        SED = pd.read_csv(path, sep=',', header=0)
        x = SED['lambda'].to_numpy()
        if 'all' in which_type.lower():
            y = SED['All'].to_numpy()
        elif 'low' in which_type.lower():
            y = SED['Low_luminosity'].to_numpy()
        elif 'mid' in which_type.lower():
            y = SED['Mid_luminosity'].to_numpy()
        elif 'high' in which_type.lower():
            y = SED['High_luminosity'].to_numpy()
        else:
            raise Exception("which_type can be 'All', 'mid', 'high', 'low'")
    elif 'wissh' in which_sed.lower():
        path = path+'wissh_sed.dat'
        if add_path:
           path = add_path+path
        SED = np.loadtxt(path, delimiter=',')
        x = np.log10(SED[:, 0])
        y = np.log10(SED[:, 1])
    elif 'richards'in which_sed.lower():
        path= path+'richards_sed.dat'
        if add_path:
           path = add_path+path
        SED = np.loadtxt(path, delimiter = ',')
        x = SED[:,0]
        if 'all' in which_type.lower():
            y = SED[:,1]
        elif 'blue' in which_type.lower():
            y = SED[:,3]
        elif 'red' in which_type.lower():
            y = SED[:,4]
        elif 'opt_lum' in which_type.lower():
            y = SED[:,5]
        elif 'opt_dim' in which_type.lower():
            y = SED[:,6]
        elif 'ir_lum' in which_type.lower():
            y = SED[:,7]
        elif 'ir_dim' in which_type.lower():
            y = SED[:,8]
        else:
            raise Exception("which_type can be 'All', 'blue', 'red', 'opt_lum', 'opt_dim', 'ir_lum', 'ir_dim' ")
        
        
    else:
        raise Exception("Which_sed can be 'wissh', 'krawczyk', 'richards' ")
    if normalization and log_log:
        normalization = [10**k for k in normalization]
        x = 10**x
        y = 10**y
        norm = normalization[1]/interpolate(x, y, normalization[0])
        y = y*norm
        x = np.log10(x)
        y = np.log10(y)
    elif normalization and not log_log:
        x = 10**x
        y = 10**y
        norm = normalization[1]/interpolate(x, y, normalization[0])
        y = y*norm
    elif not log_log:
        x = 10**x
        y = 10**y
    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    sed = np.concatenate((x, y), axis=1)

    return sed



def interpolate(x, y, x0, out_of_bounds='error', sort=True, log_log=True):
    """
    It returns the value of y computed at x0 linearly interpolating between 
    two adjacent points. x and y must have the same size.
    x :  N*1 array
    y :  N*1 array
    x0 : float
    out_of_bounds : Number, np.nan, 'extrapolate', 'error'. It determines the behaviour of the interpolation when x0 is out of bounds

    sort: logical, if true the templates are sorted befor computation

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

def get_filter_names(key='', to_print=True):
    lista=glob.glob('Tavole_utili/filtri/'+'*'+key+'*')
    lista = [i[20:-4] for i in lista]
    if to_print:
       for name in lista: print(name)
    return lista

def get_filter(filter_name, exact_name=False):
    if exact_name:
       filtro=np.loadtxt('Tavole_utili/filtri/'+filter_name+'.dat')
    else:
       lista=glob.glob('Tavole_utili/filtri/'+'*'+filter_name+'*')
    if len(lista) ==1:
       filtro = np.loadtxt(lista[0]) 
       print('Returning data of ' + lista[0][20:-4])
    elif len(lista)>1:
          print('Esiste più di un filtro')
          for name in lista: print('Returning data of ' + name[20:-4])
          filtro = [np.loadtxt(i) for i in lista] 
    elif len(lista)==0:
          raise Exception("Non esiste nessun filtro con questo nome")
 
    return filtro



def three_2_two(data, other_data=[], names =False ):
    """
    It transforms a 3-Dimesional array into a 2D table, where each row has data from 1 source.
    Table's columns give luminosities at the different bands + possibly other physical information (e.g. redshift)

    Parameters
    ----------
    data : Numpy 3D array.
           Numpy 3D array with magnitudes or luminosity
    other_data : List or tuple of arrays.
                 List (or tuple) with other infromation to append to the table.
                 E.g. other_data = [redshift, EBV, Lbol] where each feature is a 
                 NQSO x 1 array. 
       
    names : List of strings, OPTIONAL
            list of strings, containig the names of the bands ( e.g. u, g, K, W1) 
            and the names of the other features (e.g. redshift, EBV). 
            If passed, a Pandas Data Frame with columns names is returned
            

    Returns : 2D array or Pandas Data Frame
   

    """
    Nqso = data.shape[0]
    Nbands = data.shape[1]
    Ndata = data.shape[2]
    NewData = np.zeros((Nqso, Nbands*Ndata+len(other_data)))
    for i in range(Nbands):
        for k in range(Ndata):
            NewData[:, i*Ndata+k] =data[:,i,k]
    
    for i, feature in enumerate(other_data):
        NewData[:, Ndata*Nbands+i] = feature
        
    if names and Ndata ==3:
       name_array =[]
       for i,name in enumerate(names):
           if i <  Nbands:
               name_array.append('lambda_'+name)
               name_array.append(name)
               name_array.append('err_'+name)
           else:
               name_array.append(name)
       NewData = pd.DataFrame(NewData, columns = name_array)  
    elif names and Ndata ==2:
       name_array =[]
       for i,name in enumerate(names):
           if i <  Nbands:
               name_array.append(name)
               name_array.append('err_'+name)
           else:
               name_array.append(name)
        
       NewData = pd.DataFrame(NewData, columns = name_array)  
    return NewData


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
        raise exception('Data ust be a Pandas DataFrame or a numpy 2D array')
     
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


def get_luminosity(magnitudes, redshift, H0=70, Om0 =0.3, Return_Fluxes = False):         

    luminosity = np.zeros(magnitudes.shape)
    
    if len(magnitudes.shape) == 2:
        for i in range(magnitudes.shape[0]):
            luminosity[i, 1] = 10**(-0.4*(magnitudes[i,1] +48.6))*(2.998e18/magnitudes[i,0]) #magntitudes to fluxes
            luminosity[i, 2] = luminosity[i,1] * magnitudes[i, 2]*0.4*np.log(10)   #error on fluxes
            luminosity[i, 0] = magnitudes[i,0]/(redshift+1)       #rest frame wavelengths
        if not Return_Fluxes:
            dl = FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance(redshift).to(units.cm).value 
            for i in range(magnitudes.shape[0]):
                luminosity[i, 1] = luminosity[i, 1]*dl*dl*4*np.pi 
                luminosity[i, 2] = luminosity[i, 2]*dl*dl*4*np.pi
     
    elif len(magnitudes.shape) == 3:    
        for i in range(magnitudes.shape[1]):
            luminosity[:, i, 1] = 10**(-0.4*(magnitudes[:,i,1] +48.6))*(2.998e18/magnitudes[:,i,0]) #magntitudes to fluxes
            luminosity[:, i, 2] = luminosity[:,i,1] * magnitudes[:, i, 2]*0.4*np.log(10)   #error on fluxes
            luminosity[:, i, 0] = magnitudes[:,i,0]/(redshift+1)       #rest frame wavelengths
        if not Return_Fluxes:
            dl = FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance(redshift).to(units.cm).value 
            for i in range(magnitudes.shape[1]):
                 luminosity[:, i, 1] = luminosity[:, i, 1]*dl*dl*4*np.pi 
                 luminosity[:, i, 2] = luminosity[:, i, 2]*dl*dl*4*np.pi
    
    else:
        raise exception('wrong format for magnitudes')
    
    return luminosity         
    
   

def monochromatic_lum(data, wavelength, uncertainties = False, out_of_bounds = np.nan):
    
    N = np.shape(data)[0]   #Number of QSOs
    if not uncertainties:
       lum = np.zeros((N,1))
       for i in range(0,N):
           lum[i]= interpolate(data[i,:,0],data[i,:,1], wavelength, out_of_bounds=out_of_bounds)
    else:
       lum = np.zeros((N,3)) 
       for i in range(0,N):
           lum[i,0]= interpolate(data[i,:,0],data[i,:,1], wavelength, out_of_bounds=out_of_bounds)
           lum[i,1]= interpolate(data[i,:,0],data[i,:,1]-data[i,:,2], wavelength, out_of_bounds=out_of_bounds)
           lum[i,2]= interpolate(data[i,:,0],data[i,:,1]+data[i,:,2], wavelength, out_of_bounds=out_of_bounds)
    lum =lum.astype('float')
    return lum    














