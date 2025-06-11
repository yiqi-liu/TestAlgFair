"""
Functions for training model.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import util

def split_by_id(df, id_field='ptid', frac_train=.6):
    """Split the df by id_field into train/holdout deterministically.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.
    id_field : str
        Split df by this column (e.g. 'ptid').
    frac_train : float
        Fraction assigned to train. (1 - frac_train) assigned to holdout.

    Returns
    -------
    pd.DataFrame
        Data dataframe with additional column 'split' indication train/holdout

    """
    ptid = np.sort(df[id_field].unique())
    print("Splitting {:,} unique {}".format(len(ptid), id_field))

    # deterministic split
    rs = np.random.RandomState(0)
    perm_idx = rs.permutation(len(ptid))
    num_train = int(frac_train*len(ptid))

    # obtain train/holdout
    train_idx = perm_idx[:num_train]
    holdout_idx  = perm_idx[num_train:]
    ptid_train = ptid[train_idx]
    ptid_holdout  = ptid[holdout_idx]
    print(" ...splitting by patient: {:,} train, {:,} holdout ".format(
      len(ptid_train), len(holdout_idx)))

    # make dictionaries
    train_dict = {p: "train" for p in ptid_train}
    holdout_dict  = {p: "holdout"  for p in ptid_holdout}
    split_dict = {**train_dict, **holdout_dict}

    # add train/holdout split to each
    split = []
    for e in df[id_field]:
        split.append(split_dict[e])
    df['split'] = split

    return df


def get_split_predictions(df, split):
    """Get predictions for split (train/holdout).

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.
    split : str
        Name of split (e.g. 'holdout')

    Returns
    -------
    pd.DataFrame
        Subset of df with value split.

    """
    pred_split_df = df[df['split'] == split]
    pred_split_df = pred_split_df.drop(columns=['split'])
    return pred_split_df


def build_formulas(y_col, outcomes):
    """Build regression formulas for each outcome (y) ~ y_col predictor (x).

    Parameters
    ----------
    y_col : str
        Algorithm training label.
    outcomes : list
        All outcomes of interest.

    Returns
    -------
    list
        List of all regression formulas.

    """
    if 'risk_score' in y_col:
        predictors = ['risk_score_t']
    else:
        predictors = ['{}_hat'.format(y_col)]

    # build all y ~ x formulas
    all_formulas = []
    for y in outcomes:
        for x in predictors:
            formula = '{} ~ {}'.format(y, x)
            all_formulas.append(formula)
    return all_formulas


def get_r2_df(df, formulas):
    """Short summary.

    Parameters
    ----------
    df : pd.DataFrame
        Holdout dataframe.
    formulas : list
        List of regression formulas.

    Returns
    -------
    pd.DataFrame
        DataFrame of formula (y ~ x), holdout_r2, holdout_obs.

    """
    import statsmodels.formula.api as smf
    r2_list = []

    # run all OLS regressions
    for formula in formulas:
        model = smf.ols(formula, data=df)
        results = model.fit()
        r2_dict = {'formula (y ~ x)': formula,
                   'holdout_r2': results.rsquared,
                   'holdout_obs': results.nobs}
        r2_list.append(r2_dict)
    return pd.DataFrame(r2_list)


def train_lasso(train_df, holdout_df,
                x_column_names,
                y_col,
                outcomes,
                n_folds=10,
                include_race=False,
                plot=True,
                output_dir=None):
    """Train LASSO model and get predictions for holdout.

    Parameters
    ----------
    train_df : pd.DataFrame
        Train dataframe.
    holdout_df : pd.DataFrame
        Holdout dataframe.
    x_column_names : list
        List of column names to use as features.
    y_col : str
        Name of y column (label) to predict.
    outcomes : list
        All labels (Y) to predict.
    n_folds : int
        Number of folds for cross validation.
    include_race : bool
        Whether to include the race variable as a feature (X).
    plot : bool
        Whether to save the mean square error (MSE) plots.
    output_dir : str
        Path where to save results.

    Returns
    -------
    r2_df : pd.DataFrame
        DataFrame of formula (y ~ x), holdout_r2, holdout_obs.
    pred_df : pd.DataFrame
        DataFrame of all predictions (train and holdout).
    lasso_coef_df : pd.DataFrame
        DataFrame of lasso coefficients.

    """
    if not include_race:
        # remove the race variable
        x_cols = [x for x in x_column_names if x != 'race']
    else:
        # include the race variable
        if 'race' not in x_column_names:
            x_cols = x_column_names + ['race']
        else:
            x_cols = x_column_names

    # split X and y
    train_X = train_df[x_cols]
    train_y = train_df[y_col]

    # define cross validation (CV) generator
    # separate at the patient level
    from sklearn.model_selection import GroupKFold
    group_kfold = GroupKFold(n_splits=n_folds)
    # for the synthetic data, we split at the observation level ('index')
    group_kfold_generator = group_kfold.split(train_X, train_y,
                                              groups=train_df['index'])
    # train lasso cv model
    from sklearn.linear_model import LassoCV
    n_alphas = 100
    lasso_cv = LassoCV(
                       n_alphas=n_alphas,
                       cv=group_kfold_generator,
                       random_state=0,
                       max_iter=10000,
                       fit_intercept=True,
                       normalize=True)
    lasso_cv.fit(train_X, train_y)
    alpha = lasso_cv.alpha_
    train_r2 = lasso_cv.score(train_X, train_y)
    train_nobs = len(train_X)

    # plot
    if plot:
        plt.figure()
        alphas = lasso_cv.alphas_

        for i in range(n_folds):
            plt.plot(alphas, lasso_cv.mse_path_[:, i], ':', label='fold {}'.format(i))
        plt.plot(alphas, lasso_cv.mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(lasso_cv.alpha_, linestyle='--', color='k',
                    label='alpha: CV estimate')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.xlabel(r'$\alpha$')
        plt.ylabel('MSE')
        plt.title('Mean square error (MSE) on each fold predicting {}'.format(y_col))
        plt.xscale('log')

        if include_race:
            filename = 'model_lasso_{}_race.png'.format(y_col)
        else:
            filename = 'model_lasso_{}.png'.format(y_col)
        output_dir = util.create_dir(output_dir)
        output_filepath = os.path.join(output_dir, filename)
        plt.savefig(output_filepath, bbox_inches='tight', dpi=500)

    # lasso coefficients
    coef_col_name = '{}_race_coef'.format(y_col) if include_race else '{}_coef'.format(y_col)
    lasso_coef_df = pd.DataFrame({'{}_coef'.format(y_col): lasso_cv.coef_}, index=train_X.columns)

    # number of lasso features
    original_features = len(x_cols)
    n_features = len(lasso_coef_df)

    def predictions_df(x_vals, y_col, split):
        """Short summary.

        Parameters
        ----------
        x_vals : pd.DataFrame
            DataFrame of all X values.
        y_col : str
            Name of y column (label) to predict.
        split : str
            Name of split (e.g. 'holdout').

        Returns
        -------
        pd.DataFrame
            DataFrame with 'y_hat' (prediction), 'y_hat_percentile', 'split'

        """
        y_hat = lasso_cv.predict(x_vals)
        y_hat_col = '{}_hat'.format(y_col)
        y_hat_df = pd.DataFrame(y_hat, columns=[y_hat_col])
        y_hat_percentile = util.convert_to_percentile(y_hat_df, y_hat_col)

        # include column for y_hat percentile
        y_hat_percentile_df = pd.DataFrame(y_hat_percentile)
        y_hat_percentile_df.columns = ['{}_hat_percentile'.format(y_col)]

        pred_df = pd.concat([y_hat_df, y_hat_percentile_df], axis=1)
        pred_df['split'] = split

        return pred_df

    # predict in train
    train_df_pred = predictions_df(train_X, y_col, 'train')

    # predict in holdout
    holdout_X = holdout_df[x_cols]
    holdout_df_pred = predictions_df(holdout_X, y_col, 'holdout')

    # predictions
    pred_df = pd.concat([train_df_pred, holdout_df_pred])

    # r2
    holdout_Y_pred = pd.concat([holdout_df[outcomes], holdout_df_pred], axis=1)
    formulas = build_formulas(y_col, outcomes)
    r2_df = get_r2_df(holdout_Y_pred, formulas)

    return r2_df, pred_df, lasso_coef_df
