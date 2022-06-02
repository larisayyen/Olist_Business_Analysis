
from math import radians, sin, cos, asin, sqrt


def haversine_distance(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))

def return_significative_coef(model):

    # Extract p_values
    p_values = model.pvalues.reset_index()
    p_values.columns = ['variable', 'p_value']

    # Extract coef_int
    coef = model.params.reset_index()
    coef.columns = ['variable', 'coef']
    return p_values.merge(coef,
                          on='variable')\
                   .query("p_value<0.05").sort_values(by='coef',
                                                      ascending=False)
