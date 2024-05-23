import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr

var_map = {
    'tasmin': 'TMIN',
    'tasmax': 'TMAX'
}

def plot_climatology(ds, ucdb_city, urban_vicinity, variable, URBAN, obs = None):
    """
    Plot the climatological data.

    Parameters:
        ds (xr.Dataset): Dataset containing the climatological data.
        ucdb_city (gpd.GeoDataFrame): GeoDataFrame of the city boundaries.
        urban_vicinity (object): Object representing urban vicinity.
        obs (pd.DataFrame, optional): DataFrame containing observational data (default is None).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    ds_var_period_mean = ds.mean('time').compute()
    rural_mean = ds_var_period_mean[variable].where(urban_vicinity['urban_area'] == 0).mean().compute()
    data = ds_var_period_mean[variable] - rural_mean

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(20, 10))
    
    # Compute the maximum absolute value
    max_abs_value = abs(data).max().item()
    
    im1 = ax.pcolormesh(ds.lon, ds.lat, data.values,
                    cmap = 'bwr', alpha = 0.7,
                    vmin = - max_abs_value, 
                    vmax = max_abs_value)
    
    plt.colorbar(im1, ax = ax)
    
    ucdb_city.plot(ax=ax, facecolor="none", transform=proj, edgecolor="Green", linewidth=2, zorder = 1000)
    
    ax.set_title(f'ºC')
    ax.coastlines()
    
    # Overlay the cell borders and handle NaNs
    URBAN.plot_urban_borders(urban_vicinity, ax)
    
    if obs is not None:
        for index, ob in obs.iterrows():
            ax.scatter(ob['lon'], ob['lat'], c = ob.mean('time'), cmap = 'bwr', transform=proj, zorder = 100)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    return fig

def plot_time_series(ds_var, variable,urban_vicinity,time_series=None,data_squares=False,var_map=var_map):
    '''
    Plot time series data with optional urban area overlay and additional time series overlay.

    Parameters:
    ds_var (xarray.Dataset): Dataset containing the variable data.
    variable (str): Name of the variable of interest.
    urban_vicinity (xarray.Dataset): Dataset containing information about urban areas.
    time_series (list of pandas.DataFrame, optional): List of time series dataframes to overlay on the plot.
    data_squares (bool, optional): Flag indicating whether to plot individual data squares for urban and rural areas.

    Returns:
    matplotlib.figure.Figure: The plotted figure.
    '''
    rural_mean = ds_var[variable].where(urban_vicinity['urban_area'] == 0).mean(dim=[ds_var.cf['Y'].name,ds_var.cf['X'].name]).compute()
    ds_var_period_mean=ds_var.mean('time')
    data = ds_var_period_mean[variable] - rural_mean
    
    # Calculate the climatological monthly mean (mean for each month across all years)
    climatological_monthly= data.groupby('time.month').mean('time')
    climatological_monthly_mean = data.groupby('time.month').mean(dim=[ds_var.cf['Y'].name,ds_var.cf['X'].name, 'time'])
    
    # Plot the annual cycle
    fig, ax = plt.subplots(figsize=(20, 10)) 
    climatological_monthly_mean.plot(ax=ax, marker='o', color='skyblue', linestyle='-', linewidth=2)

    # Plot individual data squares for urban and rural areas if requested
    if data_squares==True:
        for i in climatological_monthly.cf['X'].values:
            for j in climatological_monthly.cf['Y'].values:
                if urban_vicinity['urban_area'].sel({urban_vicinity.cf['X'].name: i, urban_vicinity.cf['Y'].name: j}) == 1:
                    climatological_monthly.sel({ds_var.cf['X'].name:i, ds_var.cf['Y'].name:j}).plot(ax=ax, color='red', linewidth=0.5)
                elif urban_vicinity['urban_area'].sel({urban_vicinity.cf['X'].name: i, urban_vicinity.cf['Y'].name: j}) == 0:
                    climatological_monthly.sel({ds_var.cf['X'].name:i, ds_var.cf['Y'].name:j}).plot(ax=ax, color='blue', linewidth=0.5)
    
    if time_series is not None:
        var = var_map.get(variable, None)
        obs_monthly_change = []
        time_series_df = pd.concat(time_series, axis=1)
        time_series_df.index = pd.to_datetime(time_series_df.index)
        time_series_df['month'] = time_series_df.index.month
        for i in range(1, 13):
            monthly_data = time_series_df.loc[time_series_df.index.month == i].mean()
            rural_data = rural_mean[i].values
            monthly_change = monthly_data[var] - rural_data
            obs_monthly_change.append(monthly_change)
        
        plt.plot(range(1, 13), obs_monthly_change, marker='o', color='red', linestyle='-', linewidth=2) 
    
    # Customize the plot
    ax.set_title('Annual Cycle of Sample Data')
    ax.set_xlabel('Month')
    ax.set_ylabel(f'Sample Data ºC')
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    return fig