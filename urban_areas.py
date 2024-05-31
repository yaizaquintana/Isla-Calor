import xarray as xr
import glob
import numpy as np
from skimage.morphology import dilation, square
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import dilation, square
from itertools import product
import geopandas as gpd

from utils import (
    RCM_DICT,
    MODEL_DICT,
)

def load_variable(root_esgf, variable, domain, model, scenario):
    """
    Load variable data from multiple NetCDF files.

    Parameters:
    root_esgf (str): Root directory path for ESGF data.
    variable (str): Variable name to load .
    domain (str): Domain of the data.
    model (str): Climate model identifier.
    scenario (str): Emission scenario.

    Returns:
    xr.Dataset: Combined dataset containing the variable data across all found files.
    """
    files_var = glob.glob(
        f"{root_esgf}{domain}/{RCM_DICT[domain][model].split('_')[0]}/*/{scenario}/*/{RCM_DICT[domain][model].split('_')[1]}/*/day/{variable}/*/{variable}_*.nc" 
    )
    datasets = [xr.open_dataset(f) for f in np.sort(files_var)]
    ds_var = xr.concat(datasets, dim='time')
    ds_var = fix_360_longitudes(ds_var)
    return ds_var

def fix_360_longitudes(
    dataset: xr.Dataset, lonname: str = "lon"
) -> xr.Dataset:
    """
    Fix longitude values.

    Function to transform datasets where longitudes are in (0, 360) to (-180, 180).

    Parameters
    ----------
    dataset (xarray.Dataset): data stored by dimensions
    lonname (str): name of the longitude dimension

    Returns
    -------
    dataset (xarray.Dataset): data with the new longitudes
    """
    lon = dataset[lonname]
    if lon.max().values > 180 and lon.min().values >= 0:
        dataset[lonname] = dataset[lonname].where(lon <= 180, other=lon - 360)
    return dataset

def kelvin2degC(ds, variable):
    """
    Convert a variable's units from Kelvin to Celsius in an xarray Dataset.

    Parameters:
    ds (xr.Dataset): The dataset containing the variable to convert.
    variable (str): The name of the variable to convert.

    Returns:
    xr.Dataset: The dataset with the variable converted to Celsius.
    """
    if ds[variable].attrs.get('units') == 'K':
        ds[variable] = ds[variable] -273.15
        ds[variable].attrs['units'] = 'degC'
        
    return ds

def load_ucdb_city(root, city):
    """
    Load and filter a city shapefile from the Urban Centre Database (UCDB).

    Parameters:
    root (str): The root directory where the shapefile is located.
    city (str): The name of the city to load.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the filtered city shapefile.
    """
    ucdb_info = gpd.read_file(root + '/GHS_FUA_UCD/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg')
    ucdb_city = ucdb_info.query(f'UC_NM_MN =="{city}"').to_crs(crs='EPSG:4326')
    if city == 'London':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'United Kingdom']
    return ucdb_city


def fix_sftuf(
    domain : str | None = None, 
    model : str | None = None,
    ds_sftuf : xr.DataArray | None = None, 
    ds_orog : xr.DataArray | None = None, 
    ds_sftlf: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Fix some issues in urban fraction
    Parameters
    ----------
    domain : str
        CORDEX domain
    model : str
        CORDEX-CORE model (e.g. REMO/RegCM)
    ds_sftuf : xarray.DataArray 
        Urban fraction (0-1)
    ds_orog : xarray.DataArray
        Orogrhapy (m)
    ds_sftlf : xarray.DataArray
        Land-sea percentaje (%)

    Returns
    -------
    ds_sftuf : xarray.DataArray
        Fixed urban fraction
    """
    if 'iy' in ds_sftuf.dims:
        ds_sftuf = ds_sftuf.rename({"iy": "y",
                                    "jx": "x"})    
    if (domain == "EUR-11") and (model=="REMO"):
        # longitud and latitud for ds_sftuf does not match with orog/sftlf
        ds_sftuf['lon'][:] = ds_orog['lon']
        ds_sftuf['lat'][:] = ds_orog['lat']
    elif (model == "RegCM") and (domain in ["EUR-11", "CAM-22", "SAM-22", "AUS-22", "AFR-22"]):
        ds_sftuf = ds_sftuf.assign_coords(x=ds_orog.x, y=ds_orog.y)
    
    # select time = 0
    if 'time' in ds_sftuf.dims:
        ds_sftuf = ds_sftuf.isel(time = 0)

    if "urban" in ds_sftuf.var():
        ds_sftuf = ds_sftuf.rename({'urban': 'sftuf'})
        
    return ds_sftuf

def load_fixed_variables(domain, model, root_esgf, root_nextcloud):
    """
    Load fixed variable data files for a specific domain and model from given root directories.

    Parameters:
    domain (str): The domain of the data (e.g., 'Europe').
    model (str): The model identifier (e.g., 'modelA').
    root_esgf (str): The root directory for ESGF data.
    root_nextcloud (str): The root directory for Nextcloud data.

    Returns:
    tuple: A tuple containing datasets for sftuf, orog, and sftlf.
    """
    # find fixed files
    file_sftuf = glob.glob(
            f"{root_nextcloud}{model}/urbanfraction/{MODEL_DICT[model]['sftuf']}/{domain}*.nc" 
    )
    file_orog = glob.glob(
        f"{root_esgf}{domain}/{RCM_DICT[domain][model].split('_')[0]}/ECMWF-ERAINT/evaluation/*/{RCM_DICT[domain][model].split('_')[1]}/*/fx/orog/*/orog_*.nc" 
    )
    file_sftlf = glob.glob(
        f"{root_esgf}{domain}/{RCM_DICT[domain][model].split('_')[0]}/ECMWF-ERAINT/evaluation/*/{RCM_DICT[domain][model].split('_')[1]}/*/fx/sftlf/*/sftlf_*.nc" 
    )
    
    # load fixed variables
    sftuf = xr.open_dataset(file_sftuf[0])
    orog = xr.open_dataset(file_orog[0])
    sftlf = xr.open_dataset(file_sftlf[0])

    # 360 to 180
    sftuf = fix_360_longitudes(sftuf)
    orog = fix_360_longitudes(orog)
    sftlf = fix_360_longitudes(sftlf)

    return sftuf, orog, sftlf

class Urban_vicinity:
    def __init__(self, urban_th = 0.1, urban_sur_th = 0.1, orog_diff = 100, sftlf_th = 70,
                 scale = 4, lon_city = None, lat_city = None, 
                 lon_lim = None, lat_lim = None):
                     
        self.urban_th = urban_th
        self.urban_sur_th = urban_sur_th
        self.orog_diff = orog_diff
        self.sftlf_th = sftlf_th
        self.lon_city = sftlf_th
        self.scale = scale
        self.lon_city = lon_city
        self.lat_city = lat_city
        self.lon_lim = lon_lim
        self.lat_lim = lat_lim

    def crop_area_city(
        self, 
        ds : xr.DataArray | None = None,
        res : int | None = None
    ) -> xr.DataArray:
        """
        Select area around a central city point.
    
        Parameters
        ----------
        ds : xarray.DataArray 
            xarray with longitud and latitud.
        res : xarray.DataArray
            Domain resolution (e.g. 11/22).
            
        Returns
        -------
        ds : xarray.DataArray
            Cropped xarray.
        """
        # number of cells around the city
        dlon = int(111*self.lon_lim/res)
        dlat = int(111*self.lat_lim/res)
        # select point close the city
        dist = (ds['lon']-self.lon_city)**2 + (ds['lat']-self.lat_city)**2
        [ilat], [ilon] = np.where(dist == np.min(dist))
        if ds.lon.ndim == 2:
        # crop area
            ds = ds.isel(**{
            ds.cf['Y'].name: slice(ilat-dlat,ilat+dlat),
            ds.cf['X'].name : slice(ilon-dlon,ilon+dlon)
            })   
        else:
            ds = ds.isel(**{
            'lat' : slice(ilat-dlat,ilat+dlat),
            'lon' : slice(ilon-dlon,ilon+dlon)
            })  
        return ds

    def define_masks(
        self, 
        ds_sftuf : xr.DataArray | None = None, 
        ds_orog : xr.DataArray | None = None, 
        ds_sftlf: xr.DataArray | None = None
    )-> xr.DataArray:
        """
        Define masks for urban fraction, orography and land-sea mask.
    
        Parameters
        ----------
        ds_sftuf : xarray.DataArray 
            Urban fraction (0-1)
        ds_orog : xarray.DataArray
            Orogrhapy (m)
        ds_sftlf : xarray.DataArray
            Land-sea percentaje (%)
            
        Returns
        -------
        sftuf_mask : xarray.DataArray 
            Binary mask indicating urban areas with 1 and 0 for the rest.
        orog_mask : xarray.DataArray
            Binary mask indicating of orography with .
        sftlf_mask : xarray.DataArray
            Binary mask indicating sea areas.
        sftuf_mask : xarray.DataArray 
            Binary mask indicating non-urban areas with 1 and 0 for the rest.
        """
        # sftuf
        ds_sftuf = ds_sftuf["sftuf"]
        sftuf_mask = ds_sftuf > self.urban_th
        sftuf_sur_mask = ds_sftuf <= self.urban_sur_th
        
        # orog
        ds_orog = ds_orog["orog"]
        urban_elev_max = ds_orog.where(sftuf_mask).max().item()
        urban_elev_min = ds_orog.where(sftuf_mask).min().item()
        orog_mask1 = ds_orog < (self.orog_diff + urban_elev_max)
        orog_mask2 = ds_orog > (urban_elev_min - self.orog_diff)
        orog_mask = orog_mask1 & orog_mask2
        
        #sftlf
        ds_sftlf = ds_sftlf["sftlf"]
        sftlf_mask = ds_sftlf > self.sftlf_th   
        
        # Apply orog and sftlf thresholds to the urban_mask
        sftuf_mask = sftuf_mask*sftlf_mask
        sftuf_sur_mask = sftuf_sur_mask*sftlf_mask

        self.urban_elev_min = urban_elev_min 
        self.urban_elev_max = urban_elev_max 
    
        return sftuf_mask, sftuf_sur_mask, orog_mask, sftlf_mask


    def select_urban_vicinity(
        self,
        sftuf_mask : xr.DataArray | None = None,
        orog_mask : xr.DataArray | None = None,
        sftlf_mask : xr.DataArray | None = None,
        sftuf_sur_mask : xr.DataArray | None = None,
        scale: int | None = None
    ) -> xr.DataArray:
        """
        Funtion to select a number of non-urban cells based on surrounding urban areas using a dilation operation and
        excluding large water bodies and mountains.
    
        Parameters
        ----------
        sftuf_mask : xarray.DataArray 
            Binary mask indicating urban areas with 1 and 0 for the rest.
        orog_mask : xarray.DataArray
            Binary mask indicating of orography with .
        sftlf_mask : xarray.DataArray
            Binary mask indicating sea areas.
        sftuf_mask : xarray.DataArray 
            Binary mask indicating non-urban areas with 1 and 0 for the rest.
        scale : int 
            Urban-rural ratio of grid boxes.
    
        Returns
        -------
        xarray.DataArray 
            Mask of urban (1) surrounding non-urban cells (0) and the rest (NaN).
        """
        if scale is None:
            scale = self.scale
        
        data_array = xr.DataArray((~sftuf_sur_mask).astype(int))
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])
                
        urban_cells = np.sum(sftuf_mask)
        non_urban_cells = 0
        counter = 0
        while non_urban_cells <= urban_cells * scale:
            # Dilation
            dilated_data = xr.apply_ufunc(dilation, 
                                          data_array if non_urban_cells == 0 else dilated_data, 
                                          kwargs={'footprint': kernel})
            #Delete fixed variables
            dilated_data = (dilated_data * orog_mask * sftlf_mask).astype(int)
            if np.sum(dilated_data) - urban_cells == non_urban_cells:
                print(f"Warning: No more non-urban cells can be found in iteration number {counter}")
                break
            non_urban_cells = np.sum(dilated_data) - urban_cells
            counter += 1
                                
        # select surroundings cells (not rural)
        sur = sftuf_sur_mask.astype(int) - sftuf_mask.astype(int)
        sur = sur.where(sur == 0, np.nan).fillna(1)
        non_urban_mask = xr.DataArray(dilated_data.where(~sftuf_mask).fillna(0))*sur
        urban_area = sftuf_mask.astype(int).where(sftuf_mask.astype(int) == 1, np.nan)
        urban_area = urban_area.where(non_urban_mask.astype(int) == 0, 0)
        urban_area = urban_area.to_dataset(name='urban_area')
        # Add attributes
        urban_area = Urban_vicinity.netcdf_attrs(self, urban_area)
        
        return urban_area
    
    def plot_urban_borders(self, ds, ax):
        """
        Plot the borders of urban areas on a map.
    
        Parameters:
        ds (xr.Dataset): The dataset containing longitude, latitude, and urban area data.
        ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axes on which to plot.
    
        """
        lon2d = ds.lon.values
        lat2d = ds.lat.values
        # Overlay the cell borders and handle NaNs
        for i in range(len(ds.lat) - 1):
            for j in range(len(ds.lat) - 1):
                lons = [lon2d[i, j], lon2d[i, j+1], lon2d[i+1, j+1], lon2d[i+1, j], lon2d[i, j]]
                lats = [lat2d[i, j], lat2d[i, j+1], lat2d[i+1, j+1], lat2d[i+1, j], lat2d[i, j]]

                lons = lons - abs(lon2d[i, j] - lon2d[i, j+1])/2                
                lats = lats - abs(lat2d[i, j] - lat2d[i+1, j])/2
                
                data_cell = ds['urban_area'].values[i, j]

                if data_cell == 1:
                    ax.plot(lons, lats, color='red', zorder = 100, linewidth=2)
                elif data_cell == 0:
                    ax.plot(lons, lats, color='b', zorder = 1, linewidth=2)

    def plot_fixed_variables(self, ds_sftuf, ds_orog, ds_sftlf,
                             sftuf_mask, orog_mask, sftlf_mask, urban_areas = None,
                            ):
        """
        Plot fixed variables including urban fraction, orography, and land-sea mask.
    
        Parameters:
        ds_sftuf (xr.Dataset): Dataset containing urban fraction data.
        ds_orog (xr.Dataset): Dataset containing orography data.
        ds_sftlf (xr.Dataset): Dataset containing land-sea mask data.
        sftuf_mask (xr.DataArray): Mask for urban fraction.
        orog_mask (xr.DataArray): Mask for orography.
        sftlf_mask (xr.DataArray): Mask for land-sea mask.
        urban_areas (xr.Dataset, optional): Dataset containing urban area borders.
                                            Defaults to None.
    
        Returns:
        matplotlib.figure.Figure: The generated figure.
        """
        colors = ['#7C5B49', '#92716B', '#A89080', '#C0B49E', '#DACCB9', '#F5F5DC']
        custom_cmap = LinearSegmentedColormap.from_list("custom_terrain", colors)
        
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(2, 3, subplot_kw={'projection': proj}, figsize=(20, 10))
        
        im1 = axes[0, 0].pcolormesh(ds_sftuf.lon, ds_sftuf.lat,
                                    ds_sftuf["sftuf"].values,
                                    cmap='binary',
                                    vmin = np.nanmin(ds_sftuf["sftuf"]), 
                                    vmax = np.nanmax(ds_sftuf["sftuf"]))
        fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')
        axes[0, 0].set_title('Urban Fraction')
        axes[0, 0].coastlines()
        
        im2 = axes[0, 1].pcolormesh(ds_orog.lon, ds_orog.lat,
                                    ds_orog['orog'], 
                                    cmap=custom_cmap, 
                                    vmin = np.nanmin(ds_orog['orog']), 
                                    vmax = np.nanmax(ds_orog['orog'])
        )
        fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')
        axes[0, 1].set_title('Orography')
        axes[0, 1].coastlines()
        
        im3 = axes[0, 2].pcolormesh(ds_sftlf.lon, ds_sftlf.lat,
                                    ds_sftlf["sftlf"],
                                    cmap='winter',
                                    vmin = np.nanmin(ds_sftlf["sftlf"]), 
                                    vmax = np.nanmax(ds_sftlf["sftlf"]))
        fig.colorbar(im3, ax=axes[0, 2], orientation='vertical')
        axes[0, 2].set_title('Land-sea')
        axes[0, 2].coastlines()
    
        # masks
        im1 = axes[1, 0].pcolormesh(ds_sftuf.lon, ds_sftuf.lat,
                                    ds_sftuf["sftuf"].where(sftuf_mask == 1, np.nan),
                                    cmap='binary',
                                    vmin = np.nanmin(ds_sftuf["sftuf"]), 
                                    vmax = np.nanmax(ds_sftuf["sftuf"]))
        fig.colorbar(im1, ax=axes[1, 0], orientation='vertical')
        if not urban_areas:
            axes[1, 0].set_title('Urban Fraction\n(sftuf >' +  str(self.urban_th) + ')')
        else:
            axes[1, 0].set_title('Urban Fraction\n(sftuf >' +  str(self.urban_th) + 
                                 ', Non-urban(sftuf) <= ' + str(self.urban_sur_th) + ', scale = ' + str(self.scale) +  ')')
        axes[1, 0].coastlines()
        
        im2 = axes[1, 1].pcolormesh(ds_orog.lon, ds_orog.lat,
                                    ds_orog['orog'].where(orog_mask == 1, np.nan), 
                                    cmap=custom_cmap, 
                                    vmin = np.nanmin(ds_orog['orog']), 
                                    vmax = np.nanmax(ds_orog['orog'])
        )
        fig.colorbar(im2, ax=axes[1, 1], orientation='vertical')
        axes[1, 1].set_title('Orography\n(' + str(int(self.urban_elev_min) - self.orog_diff) + 
                             ' m < orog > ' +  
                             str(int(self.urban_elev_max) + self.orog_diff) + ' m)'
                            )
        axes[1, 1].coastlines()
        
        im3 = axes[1, 2].pcolormesh(ds_sftlf.lon, ds_sftlf.lat,
                                    ds_sftlf["sftlf"].where(sftlf_mask == 1, np.nan),
                                    cmap='winter',
                                    vmin = np.nanmin(ds_sftlf["sftlf"]), 
                                    vmax = np.nanmax(ds_sftlf["sftlf"]))
        fig.colorbar(im3, ax=axes[1, 2], orientation='vertical')
        axes[1, 2].set_title('Land-sea\n(sftlf >' + str(self.sftlf_th) + '%)')
        axes[1, 2].coastlines()

        if urban_areas:
            for k in range(3):
                Urban_vicinity.plot_urban_borders(self, urban_areas, axes[1, k])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust vertical and horizontal space
        return fig

    def netcdf_attrs(self, ds):        
        """
        Add metadata to urban area file.
    
        Parameters
        ----------
        urban_area : xarray.Dataset 
            Binary mask indicating urban areas (1), non-urban (vicinity) areas (0) and NaN for the rest.
        """
        # add attribtes
        ds['urban_area'].attrs['long_name'] = 'Urban vs. vicinity. 1 corresponds to urban areas and 0 to the surrounding areas'
        
        attrs_list = ["urban_th", "urban_sur_th", "orog_diff", "sftlf_th", "sftlf_th", "scale", 
                    "lon_city", "lat_city", "lon_lim", "lat_lim"]
        
        for attr in attrs_list:
            ds['urban_area'].attrs[attr] = getattr(self, attr)
            
        return ds