import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

var_map = {
    'tasmin': 'TMIN',
    'tasmax': 'TMAX'
}

def load_ghcnd_stations(lon, lat):
    '''
    Load GHCND stations near a specific location.

    Parameters:
    lon (float): Longitude of the selected city.
    lat (float): Latitude of the selected city.

    Returns:
    gpd.GeoDataFrame: Geospatial DataFrame of nearby GHCND stations.
    '''
    ghcnd_stations_url = 'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/doc/ghcnd-stations.txt'
    ghcnd_stations_column_names = ['code', 'lat', 'lon', 'elev', 'name', 'net', 'numcode']
    ghcnd_stations_column_widths = [   11,     9,    10,      7,     34,     4,       10 ]
    df = pd.read_fwf(ghcnd_stations_url, header = 0, widths = ghcnd_stations_column_widths, names = ghcnd_stations_column_names)
    ghcnd_stations=gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs = 'EPSG:4326')
    rval = ghcnd_stations.assign(dist = ghcnd_stations.distance(Point(lon, lat)))
    rval.sort_values(by = 'dist', inplace = True)
    rval = rval[rval.dist < 0.5].to_crs(epsg=3857)
    return rval

def get_ghcnd_df(code):
    '''
    Load GHCND data for a specific station.

    Parameters:
    code (str): The station code.

    Returns:
    pd.DataFrame: DataFrame containing the GHCND data for the specified station.
    '''
    baseurl = '/lustre/gmeteo/WORK/WWW/chus/ghcnd/data'
    try:
      rval = pd.read_csv(f'{baseurl}/{code[0]}/{code}.csv.gz',
        compression = 'gzip',
        index_col = 'DATE',
        parse_dates = True,
        low_memory = False # Avoid warning on mixed dtypes for some (unused) columns
      )
    except:
      print(f'Problem downloading {code}')
      rval = pd.DataFrame()
    return(rval)

def get_valid_timeseries(city, stations, ds_var, variable = 'tasmin', valid_threshold=0.8, idate='1979-01-01', fdate='2014-12-31',var_map=var_map):
    '''
    Retrieves valid time series data for a specific variable from GHCND stations for a given city.

    Parameters:
    city (str): The name of the city for which the data is to be retrieved.
    stations (GeoDataFrame): A GeoDataFrame containing station metadata.
    var (str): The variable of interest (default is 'PRCP' for precipitation).
    valid_threshold (float): The threshold proportion of valid records required (default is 0.8).
    idate (str): The start date for the period of interest (default is '1979-01-01').
    fdate (str): The end date for the period of interest (default is '2014-12-31').
    var_map (dict): A dictionary mapping the variable names from the input to the dataset variable names.
    
    Returns:
    tuple: A tuple containing:
        - GeoDataFrame: The subset of stations with valid data.
        - list: A list of valid time series data.
        - xr.Dataset: The subset of the dataset containing the selected period.
    '''
    var = var_map.get(variable, None)
    period = slice(idate, fdate)
    ds_var_period=ds_var.sel(time=period)
    ndays = (pd.to_datetime(fdate)-pd.to_datetime(idate)).days
    valid_codes = []
    valid_time_series = []
    for stn_code in stations.code:
        stn_data = get_ghcnd_df(stn_code)
        if stn_data.empty:
            continue
        availvars = available_vars(stn_data)
        if var in availvars:
          valid_records = stn_data[var].loc[period].notna().sum()/ndays
          if valid_records > valid_threshold:
            print(f'{city} -- {stn_data.NAME[0]} - {var} has {100*valid_records:.1f}% valid records in {idate} to {fdate}')
            valid_codes.append(stn_code)
            valid_time_series.append({'data':stn_data[var].loc[period]/10.0,'code':stn_code})
  
    return(stations[stations.code.isin(valid_codes)], valid_time_series, ds_var_period)

def available_vars(station):
    """
    Determines which variables are available in the station's dataset.

    Parameters:
    station (DataFrame): The DataFrame containing the station's data.

    Returns:
    set: A set of available variables that intersect with the known set of variables.
    """
    return(set(station.columns).intersection({'PRCP', 'TAVG', 'TMAX', 'TMIN', 'SNWD'}))

def get_season(ds_var_period, time_series, season='all'):
    '''
    Selects the chosen season from the observations and dataset.

    Parameters:
    ds_var_period (xarray.Dataset): The dataset containing the variable data over the desired period.
    time_series (list of pd.Series): List of time series data from observations.
    season (str): The season to select ('all', 'DJF', 'MAM', 'JJA', 'SON').

    Returns:
    xarray.Dataset: The subset of the dataset containing the selected season.
    pd.DataFrame: DataFrame of the time series data for the selected season.
    '''
    if season == 'all':
        # Return the whole period if season is 'all'
        ds_season = ds_var_period
        ts_season_list = []
        for item in time_series:
            time_series_df = pd.DataFrame(item['data'])
            station_code = item['code']  # Get the station code
            ts_season = {}
            ts_season['data'] = time_series_df
            ts_season['code'] = station_code  # Add station code to the DataFrame
            ts_season_list.append(ts_season)
            
    else:
        # Select the months corresponding to each season
        seasons = {
            'DJF': [12, 1, 2],
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11]
        }
        selected_months = seasons.get(season, None)

        
        # Select the data for the chosen season from the dataset
        ds_season = ds_var_period.sel(time=ds_var_period['time.month'].isin(selected_months))
        
        # Select the data for the chosen season from the time series
        ts_season_list = []
        for item in time_series:
            time_series_df = pd.DataFrame(item['data'])
            time_series_df.index = pd.to_datetime(time_series_df.index)
            time_series_df['month'] = time_series_df.index.month
            station_code = item['code']  # Get the station code
            ts_season = {}
            ts_season['data'] = time_series_df.loc[time_series_df['month'].isin(selected_months)].drop(columns=['month'])
            ts_season['code'] = station_code  # Add station code to the DataFrame
            ts_season_list.append(ts_season)

    return ds_season, ts_season_list


