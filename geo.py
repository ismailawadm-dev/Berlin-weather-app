import numpy as np

def bbox_from_center(lat, lon, dlat=1.5, dlon=2.0):
    return (lon-dlon, lat-dlat, lon+dlon, lat+dlat)

def nearest_idx(lat_arr, lon_arr, lat, lon):
    i = np.abs(lat_arr - lat).argmin()
    j = np.abs(lon_arr - lon).argmin()
    return int(i), int(j)
