##############################################################
############# 9. Clustergrammer
##############################################################

def clustergram_test(dataframe, filter_rows=True, filter_rows_by='var', filter_rows_n=500, normalize=True, col_categories=None):
    data_norm = normalize_data(dataframe)

    # Initialize Clustergrammer network
    net = Network(clustergrammer_widget)
    net.load_df(data_norm)
    net.normalize()

    # If column categories have been specified, add them
    if col_categories != None:
      cat_data = ...
      ####### Code to convert a sample characteristics dataframe
      net.add_cats(cat_data)

    net.cluster()
    return net.widget()