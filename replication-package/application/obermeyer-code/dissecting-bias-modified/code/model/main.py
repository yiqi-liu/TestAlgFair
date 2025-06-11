"""
Main script to train lasso model and save predictions.
"""
import pandas as pd
import numpy as np
import os

import features
import model
import util


def load_data_df():
    """Load data dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame to use for analysis.

    """
    # define filepath
    git_dir = util.get_git_dir()
    data_fp = os.path.join(git_dir, 'data', 'data_new.csv')

    # load df
    data_df = pd.read_csv(data_fp)

    # because we removed patient
    data_df = data_df.reset_index()
    return data_df


def get_Y_x_df(df, verbose):
    """Get dataframe with relevant x and Y columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.
    verbose : bool
        Print statistics of features.

    Returns
    -------
    all_Y_x_df : pd.DataFrame
        Dataframe with x (features) and y (labels) columns
    x_column_names : list
        List of all x column names (features).
    Y_predictors : list
        All labels (Y) to predict.

    """
    # cohort columns
    cohort_cols = ['index']

    # features (x)
    x_column_names = features.get_all_features(df, verbose)

    # include log columns
    df['log_cost_t'] = util.convert_to_log(df, 'cost_t')
    df['log_cost_avoidable_t'] = util.convert_to_log(df, 'cost_avoidable_t')

    # labels (Y) to predict
    Y_predictors = ['log_cost_t', 'gagne_sum_t', 'log_cost_avoidable_t']

    # redefine 'race' variable as indicator
    df['dem_race_black'] = np.where(df['race'] == 'black', 1, 0)

    # additional metrics used for table 2 and table 3
    table_metrics = ['dem_race_black', 'risk_score_t', 'program_enrolled_t',
                     'cost_t', 'cost_avoidable_t']

    # combine all features together -- this forms the Y_x df
    all_Y_x_df = df[cohort_cols + x_column_names + Y_predictors + table_metrics].copy()

    return all_Y_x_df, x_column_names, Y_predictors


def main():
    # load data
    data_df = load_data_df()

    # subset to relevant columns
    all_Y_x_df, x_column_names, Y_predictors = get_Y_x_df(data_df, verbose=True)

    # assign to 2/3 train, 1/3 holdout
    all_Y_x_df = model.split_by_id(all_Y_x_df, id_field='index',
                                   frac_train=.67)

    # define train, holdout
    # reset_index for pd.concat() along column
    train_df = all_Y_x_df[all_Y_x_df['split'] == 'train'].reset_index(drop=True)
    holdout_df = all_Y_x_df[all_Y_x_df['split'] == 'holdout'].reset_index(drop=True)

    # define output dir to save results
    git_dir = util.get_git_dir()
    OUTPUT_DIR = util.create_dir(os.path.join(git_dir, 'results'))

    # define parameters
    include_race = False
    n_folds = 10
    save_plot = True
    save_r2 = True

    # train model with Y = 'log_cost_t'
    log_cost_r2_df, \
    pred_log_cost_df, \
    log_cost_lasso_coef_df = model.train_lasso(train_df,
                                               holdout_df,
                                               x_column_names,
                                               y_col='log_cost_t',
                                               outcomes=Y_predictors,
                                               n_folds=n_folds,
                                               include_race=include_race,
                                               plot=save_plot,
                                               output_dir=OUTPUT_DIR)

    # train model with Y = 'gagne_sum_t'
    gagne_sum_t_r2_df, \
    pred_gagne_sum_t_df, \
    gagne_sum_t_lasso_coef_df = model.train_lasso(train_df,
                                                  holdout_df,
                                                  x_column_names,
                                                  y_col='gagne_sum_t',
                                                  outcomes=Y_predictors,
                                                  n_folds=n_folds,
                                                  include_race=include_race,
                                                  plot=save_plot,
                                                  output_dir=OUTPUT_DIR)

    # train model with Y = 'log_cost_avoidable_t'
    log_cost_avoidable_r2_df, \
    pred_log_cost_avoidable_df, \
    log_cost_avoidable_lasso_coef_df = model.train_lasso(train_df,
                                                         holdout_df,
                                                         x_column_names,
                                                         y_col='log_cost_avoidable_t',
                                                         outcomes=Y_predictors,
                                                         n_folds=n_folds,
                                                         include_race=include_race,
                                                         plot=save_plot,
                                                         output_dir=OUTPUT_DIR)

    if save_r2:
        formulas = model.build_formulas('risk_score_t', outcomes=Y_predictors)
        risk_score_r2_df = model.get_r2_df(holdout_df, formulas)

        r2_df = pd.concat([risk_score_r2_df,
                           log_cost_r2_df,
                           gagne_sum_t_r2_df,
                           log_cost_avoidable_r2_df])

        # save r2 file CSV
        if include_race:
            filename = 'model_r2_race.csv'
        else:
            filename = 'model_r2.csv'
        output_filepath = os.path.join(OUTPUT_DIR, filename)
        print('...writing to {}'.format(output_filepath))
        r2_df.to_csv(output_filepath, index=False)

    def get_split_predictions(df, split):
        pred_split_df = df[df['split'] == split]
        pred_split_df = pred_split_df.drop(columns=['split'])
        return pred_split_df

    # get holdout predictions
    holdout_log_cost_df = get_split_predictions(pred_log_cost_df,
                                                split='holdout')
    holdout_gagne_sum_t_df = get_split_predictions(pred_gagne_sum_t_df,
                                                   split='holdout')
    holdout_log_cost_avoidable_df = get_split_predictions(pred_log_cost_avoidable_df,
                                                          split='holdout')

    holdout_pred_df = pd.concat([holdout_df, holdout_log_cost_df,
                                 holdout_gagne_sum_t_df,
                                 holdout_log_cost_avoidable_df], axis=1)

    holdout_pred_df_subset = holdout_pred_df[['index', 'dem_race_black',
                                              'risk_score_t', 'gagne_sum_t',
                                              'cost_t', 'cost_avoidable_t',
                                              'program_enrolled_t',
                                              'log_cost_t_hat',
                                              'gagne_sum_t_hat',
                                              'log_cost_avoidable_t_hat']].copy()

    # add risk_score_percentile column
    holdout_pred_df_subset['risk_score_t_percentile'] = \
        util.convert_to_percentile(holdout_pred_df_subset, 'risk_score_t')

    # save to CSV
    if include_race:
        filename = 'model_lasso_predictors_race.csv'
    else:
        filename = 'model_lasso_predictors.csv'
    output_filepath = os.path.join(OUTPUT_DIR, filename)
    print('...HOLDOUT PREDICTIONS saved to {}'.format(output_filepath))
    holdout_pred_df_subset.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    main()
