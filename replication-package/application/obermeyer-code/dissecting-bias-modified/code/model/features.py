"""
Functions for creating features.
"""


def get_dem_features(df):
    """Get demographic features.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.

    Returns
    -------
    list
        List of demographic features.

    """
    dem_features = []
    prefix = 'dem_'
    for col in df.columns:
        if prefix == col[:len(prefix)]:
            if 'race' not in col:
                dem_features.append(col)
    return dem_features


def get_comorbidity_features(df):
    """Get comorbidity features.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.

    Returns
    -------
    list
        List of comorbidity features.

    """
    comorbidity_features = []
    comorbidity_sum = 'gagne_sum_tm1'
    suffix_elixhauser = '_elixhauser_tm1'
    suffix_romano = '_romano_tm1'

    for col in df.columns:
        if col == comorbidity_sum:
            comorbidity_features.append(col)
        elif suffix_elixhauser == col[-len(suffix_elixhauser):]:
            comorbidity_features.append(col)
        elif suffix_romano == col[-len(suffix_romano):]:
            comorbidity_features.append(col)
        else:
            continue
    return comorbidity_features


def get_cost_features(df):
    """Get cost features.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.

    Returns
    -------
    list
        List of cost features.

    """
    cost_features = []
    prefix = 'cost_'
    for col in df.columns:
        if prefix == col[:len(prefix)]:
            # 'cost_t', 'cost_avoidable_t' are outcomes, not a features
            if col not in ['cost_t', 'cost_avoidable_t']:
                cost_features.append(col)
    return cost_features


def get_lab_features(df):
    """Get lab features.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.

    Returns
    -------
    list
        List of lab features.

    """
    lab_features = []
    suffix_labs_counts = '_tests_tm1'
    suffix_labs_low = '-low_tm1'
    suffix_labs_high = '-high_tm1'
    suffix_labs_normal = '-normal_tm1'
    for col in df.columns:
        # get lab features
        if suffix_labs_counts == col[-len(suffix_labs_counts):]:
            lab_features.append(col)
        elif suffix_labs_low == col[-len(suffix_labs_low):]:
            lab_features.append(col)
        elif suffix_labs_high == col[-len(suffix_labs_high):]:
            lab_features.append(col)
        elif suffix_labs_normal == col[-len(suffix_labs_normal):]:
            lab_features.append(col)
        else:
            continue
    return lab_features


def get_med_features(df):
    """Get med features.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.

    Returns
    -------
    list
        List of med features.

    """
    med_features = []
    prefix = 'lasix_'
    for col in df.columns:
        if prefix == col[:len(prefix)]:
            med_features.append(col)
    return med_features


def get_all_features(df, verbose=False):
    """Get all features.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.
    verbose : bool
        Print statistics of features.

    Returns
    -------
    list
        List of all features.

    """
    dem_features = get_dem_features(df)
    comorbidity_features = get_comorbidity_features(df)
    cost_features = get_cost_features(df)
    lab_features = get_lab_features(df)
    med_features = get_med_features(df)

    x_column_names = dem_features + comorbidity_features + cost_features + \
                     lab_features + med_features

    if verbose:
        print('Features breakdown:')
        print('   {}: {}'.format('demographic', len(dem_features)))
        print('   {}: {}'.format('comorbidity', len(comorbidity_features)))
        print('   {}: {}'.format('cost', len(cost_features)))
        print('   {}: {}'.format('lab', len(lab_features)))
        print('   {}: {}'.format('med', len(med_features)))
        print(' {}: {}'.format('TOTAL', len(x_column_names)))

    return x_column_names
