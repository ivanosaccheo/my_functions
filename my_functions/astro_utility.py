import os
import numpy as np 
import pandas as pd
import requests
from scipy.interpolate import interp1d
from astropy import units
from astropy.cosmology import FlatLambdaCDM
from astropy import units, constants
from my_functions import library as lb


PATH_TO_DATA = lb.PATH_TO_DATA 

def get_flux(magnitudes):

    magnitudes = np.asarray(magnitudes)

    if magnitudes.ndim not in (2, 3):
        raise ValueError("Input must be (Nbands,3) or (Nsources,Nbands,3)")

    if magnitudes.shape[-1] != 3:
        raise ValueError("Last dimension must be 3: [lambda, mag, err_mag]")

    fluxes = np.zeros_like(magnitudes, dtype = float)
    wavlen = magnitudes[..., 0]
    mag = magnitudes[..., 1]
    err = magnitudes[..., 2]

    flux = 10**(-0.4*(mag +48.6))*(2.998e18/wavlen)
    err_flux = flux * err * 0.4*np.log(10)
    
    fluxes[..., 0] = wavlen
    fluxes[..., 1] = flux
    fluxes[..., 2] = err_flux

    return fluxes       


def get_luminosity(magnitudes, redshift, H0 = 70, Om0 = 0.3):  
    luminosity = get_flux(magnitudes)
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    dl = cosmo.luminosity_distance(redshift).cgs.value 
    factor = (dl*dl*4*np.pi)

    if luminosity.ndim ==2:
        luminosity[:, 0] = luminosity[:, 0] / (redshift+1)
        luminosity[:, 1:] = factor * luminosity[:, 1:]
    else:
        luminosity[..., 0] = luminosity[..., 0] / (redshift[:,None]+1)
        luminosity[..., 1:] = factor[:, None, None] * luminosity[..., 1:]

    return luminosity


def get_magnitudes(luminosity, redshift, H0 =70, Om0 = 0.3):
    luminosity = np.asarray(luminosity)

    if luminosity.ndim not in (2, 3):
        raise ValueError("Input must be (Nbands,3) or (Nsources,Nbands,3)")
    if luminosity.shape[-1] != 3:
        raise ValueError("Last dimension must be 3: [lambda, mag, err_mag]")
    
    magnitudes = np.zeros_like(luminosity)
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    dl = cosmo.luminosity_distance(redshift).cgs.value 
    
    if luminosity.ndim == 2:
        magnitudes[:,0] = luminosity[:,0]*(redshift + 1)
        magnitudes[:,1:] = luminosity[:,1:]/(dl*dl*4*np.pi)
    
        magnitudes[:,1:] = magnitudes[:,1:]*magnitudes[:,0][:,None]/2.998e18  #Fnu

        magnitudes[:,2] = 2.5*(magnitudes[:,2]/magnitudes[:,1])/np.log(10)
        magnitudes[:,1] = -2.5*np.log10( magnitudes[:,1]) -48.6

    elif luminosity.ndim == 3: 
        
        magnitudes[:,:,0] = luminosity[:,:,0] * (redshift + 1)
        magnitudes[:,:,1:] = luminosity[:,:,1:] / (4*np.pi*dl[:,None]**2)
        magnitudes[:,:,1:] = magnitudes[:,:,1:]*magnitudes[:,:,0][:,:,None] / 2.998e18

        magnitudes[:,:,2] = 2.5 * (magnitudes[:,:,2] / magnitudes[:,:,1]) / np.log(10)
        magnitudes[:,:,1] = -2.5*np.log10(magnitudes[:,:,1]) - 48.6
   
    return magnitudes


def get_monochromatic_lum(data, wavelength, uncertainties = False,  out_of_bounds = np.nan):

    def interp_loglog(x, y, x0, out_of_bounds):
        select = np.isfinite(y) & np.isfinite(x)
        f = interp1d(x[select], y[select], fill_value = out_of_bounds, bounds_error = False)
        return 10**f(x0)
    
    
    wavelength = np.atleast_1d(wavelength)

    log_x = np.log10(data[...,0])        
    log_y = np.log10(data[...,1])         
    log_wl = np.log10(wavelength)     
    
    if uncertainties:
       log_y_low = np.log10(data[...,1] - data[...,2])
       log_y_up =  np.log10(data[...,1] + data[...,2])

    if data.ndim == 2: # single object
        lum= interp_loglog(log_x, log_y, log_wl , out_of_bounds = out_of_bounds)
        if uncertainties: 
            lum_low = interp_loglog(log_x, log_y_low,log_wl, out_of_bounds=out_of_bounds)
            lum_up =  interp_loglog(log_x, log_y_up, log_wl, out_of_bounds=out_of_bounds)
            lum = np.stack([lum, lum_low, lum_up], axis = -1)
    
    elif data.ndim == 3:
        lum = np.array([interp_loglog(x,y, log_wl, out_of_bounds = out_of_bounds) for x, y  in zip(log_x, log_y)])
        if uncertainties: 
            lum_low =np.array([interp_loglog(x,y, log_wl, out_of_bounds = out_of_bounds) for x, y  in zip(log_x, log_y_low)])
            lum_up = np.array([interp_loglog(x,y, log_wl, out_of_bounds = out_of_bounds) for x, y  in zip(log_x, log_y_up)])
            lum = np.stack([lum, lum_low, lum_up], axis = -1)
    return lum.squeeze() 


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
        numeratore = np.trapezoid(f_lambda_filter*self.transmission[:,1]*self.transmission[:,0], 
                              self.transmission[:,0])/2.998e18
        denominatore = np.trapezoid(self.transmission[:,1]/self.transmission[:,0], self.transmission[:,0])
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

def get_host():
     path = os.path.join(PATH_TO_DATA, path)
     return pd.read_csv(path, header = 0, sep = ',' ).to_numpy()



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


def sersic_get_bn(n):
    return 2*n - 1./3 + (4./405)*(n**(-1)) + (46./25515)*(n**(-2)) + (131./1148175)*(n**(-3)) - (2194697./30690717750)*(n**(-4))

def sersic_L_total(n, Ie = 1, Re =1):
    """Returns the total luminosity of a n-Sersic profile with Intensity Ie 
       at the effective radius Re. 
       From Graham 2005"""
    from scipy.special import gamma, gammainc
    bn = sersic_get_bn(n)
    Ltot = (2*np.pi * n) * (Ie * Re*Re) * (np.exp(bn)/bn**(2*n))*gamma(2*n)
    return Ltot
    
def sersic_L_r(n, r, Ie = 1, Re =1):
    """Returns the luminosity integrated between 0 and r of a n-Sersic profile with Intensity Ie 
       at the effective radius Re. 
       From Graham 2005"""
    from scipy.special import gamma, gammainc
    bn = sersic_get_bn(n)
    x = bn*(r/Re)**(1/n)
    Lr = (2*np.pi * n) * (Ie * Re*Re) * (np.exp(bn)/bn**(2*n))*gammainc(2*n, x)*gamma(2*n)
    return Lr


def send_request_IVOA(filter_id):
    url = f"https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php?ID={filter_id}"
    r = requests.get(url)
    r.raise_for_status()
    return r.content


def get_filter_votable(filter_id):
    from astropy.io.votable import parse
    import io 
    result = send_request_IVOA(filter_id)
    votable = parse(io.BytesIO(result))
    return votable


def get_filter_from_IVOA(filter_id):
    """Returns effective wavelength and transmission curve for a filter retrieved
    from the IVOA filter profile service"""
    votable = get_filter_votable(filter_id)
    try:
        wav_eff = votable.get_field_by_id("WavelengthEff").value
        transmission = votable.get_first_table().to_table().to_pandas()
        return wav_eff, transmission
    except KeyError as e:
        print(f"The request for {filter_id} returned an empty votable. Check the spell")
        raise e

def add_filter_from_IVOA(filter_id, overwrite = False):
    """Retrieves and saves filter from the IVOA filter profile system. The filter can then be accessed with the 
    filtro() method"""
    filepath = os.path.join(lb.PATH_TO_DATA, "tables/filters")
    filter_name = filter_id.replace("/","_")
    existing_names = [i for i in os.listdir(filepath) if i.endswith('.dat')]
    matching_names = [i for i in existing_names if filter_name.casefold() in i.casefold()]
    already_available = len(matching_names)>0
    if already_available and (not overwrite):
        print(f"{filter_name} is already saved. If you want to overwrite it set 'overwrite' = True")
        return 
    wav_eff, transmission = get_filter_from_IVOA(filter_id)
    wav_eff_table = pd.read_csv(os.path.join(filepath, "filter_list.txt"), sep ="\s+")
    if already_available:
        mask = wav_eff_table["Name"].str.casefold() == filter_name.casefold()
        wav_eff_table.loc[mask, "eff_wavelength"] = wav_eff
    else:
        new_row = pd.DataFrame(data = {"Name" : [filter_name], "eff_wavelength" : [wav_eff]})
        wav_eff_table = pd.concat([wav_eff_table, new_row], ignore_index = True)
    wav_eff_table.to_csv(os.path.join(filepath, "filter_list.txt"), sep = " ")
    transmission.to_csv(os.path.join(filepath, f"{filter_name}.dat"), sep =" ", index = False, header = False)

