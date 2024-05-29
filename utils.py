RCM_DICT = {
    'EUR-11': { 
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'EUR-22': {
        'REMO': 'GERICS_REMO2015',
    },
    'WAS-22': {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'EAS-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'CAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'SAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'NAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'AUS-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'AFR-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
}


MODEL_DICT={
    'REMO' : dict(sftuf='orig_v3', orog='orog',sftlf='sftlf'),
    'RegCM' : dict(sftuf='', orog='orog',sftlf='sftlf'),
}

# Dictionary containing city locations and their respective domains
location = {
     'Mexico City' : dict(lon=-99.0833, lat=19.4667, domain = 'CAM-22'),
     'Buenos Aires' : dict(lon=-58.416, lat=-34.559, domain = 'SAM-22'),
     'New York' : dict(lon=-74.2261, lat=40.8858, domain = 'NAM-22'),
     'Sydney' : dict(lon=151.01810, lat=-33.79170, domain = 'AUS-22'),
     'Beijing' : dict(lon=116.41, lat=39.90, domain = 'EAS-22'),
     'Tokyo' : dict(lon = 139.84, lat = 35.65, domain = 'EAS-22'),
     'Jakarta' : dict(lon = 106.81, lat = -6.2, domain = 'SEA-22'), 
     'Johannesburg' : dict(lon=28.183, lat=-25.733, domain = 'AFR-22'),
     'Riyadh' : dict(lon=46.73300, lat=24.7000, domain = 'WAS-22'),
     'Berlin' : dict(lon=13.4039, lat=52.4683, domain = 'EUR-11'),
     'Paris' : dict(lon=  2.35, lat=48.85, domain = 'EUR-11'),
     'London' : dict(lon= -0.13, lat=51.50, domain = 'EUR-11'),
     'Madrid' : dict(lon= -3.70, lat=40.42, domain = 'EUR-11'),
     'Los Angeles': dict(lon = -118.24, lat = 34.05, domain = 'NAM-22'),
     'Montreal': dict(lon = -73.56, lat = 45.50, domain = 'NAM-22'),
     'Chicago': dict(lon = -87.55, lat = 41.73, domain = 'NAM-22'),
     'Bogota': dict(lon = -74.06, lat = 4.62, domain = 'SAM-22'),
     'Baghdad': dict(lon = 44.40, lat = 33.34, domain = 'WAS-22'),
     'Tehran': dict(lon = 51.42, lat = 35.69, domain = 'WAS-22'),
     'Tashkent': dict(lon = 69.24, lat = 41.31, domain = 'WAS-22'),
     'Cairo': dict(lon = 31.25, lat = 30.06, domain = 'AFR-22'),
     'Delhi [New Delhi]': dict(lon = 77.22, lat = 28.64, domain = 'WAS-22'),
    'Barcelona': dict(lon = 2.18, lat = 41.39, domain = 'EUR-11'),
    'Rome': dict(lon =  12.51, lat = 41.89, domain = 'EUR-11'),
    'Athens': dict(lon =   23.72, lat =  37.98, domain = 'EUR-11'),

}
