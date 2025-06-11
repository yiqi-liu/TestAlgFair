"""
Build Table 2: performance of predictors trained on alternative labels.
"""
import pandas as pd
import os
import model.util as util


def get_concentration_metric_df(k, holdout_pred_df,
                                y_predictors=['log_cost_t',
                                              'log_cost_avoidable_t',
                                              'gagne_sum_t'],
                                outcomes=['log_cost_t', 'log_cost_avoidable_t',
                                          'gagne_sum_t', 'dem_race_black']):
    """Calculate concentration of a given outcome of interest (columns) for
    each algorithm trained label, and calculate fraction black in the high-risk
    patient group.

    Parameters
    ----------
    k : float
        Top k% patients in terms of predicted risk.
    holdout_pred_df : pd.DataFrame
        Predictions for holdout set.
    y_predictors : list
        List of algorithm training label.
    outcomes : list
        List of given outcome of interest.

    Returns
    -------
    pd.DataFrame
        Concentration metric for holdout_pred_df.
    """
    # define lookup for human readable headings in Table 2
    OUTCOME_DICT = {
        'cost_t': 'Total costs',
        'log_cost_t': 'Total costs',
        'cost_avoidable_t': 'Avoidable costs',
        'log_cost_avoidable_t': 'Avoidable costs',
        'gagne_sum_t': 'Active chronic conditions',
        'dem_race_black': 'Race black'
    }

    top_k = int(k * len(holdout_pred_df))
    all_concentration_metric = []  # save all rows of Table 2 to variable

    # iterate through each predictor (algorithm training label)
    # (this is each row in Table 2)
    for y_col in y_predictors:
        # get the predictions column name for y_col
        y_hat_col = '{}_hat'.format(y_col)

        # sort by y_hat_col
        holdout_pred_df = holdout_pred_df.sort_values(by=y_hat_col, ascending=False)
        # get top k% in terms of predicted risk
        top_k_df = holdout_pred_df.iloc[:top_k]

        # define dict to store calculated metrics for given y_col/predictor
        # (each addition to the dict appends a column from Table 2)
        concentration_dict = {
            'predictor': OUTCOME_DICT[y_col]
        }

        # iterate through each outcome
        # (concentration / frac black in highest-risk patients)
        # (this is each column in Table 2)
        for outcome in outcomes:
            if 'log_' in outcome:
                # for the outcomes presented on a log scale,
                # we sum the un-logged values.
                outcome = outcome[len('log_'):]

            # define numerator of concentration metric:
            # sum the top k of outcome
            top_k_outcome = top_k_df[outcome].sum()

            # define denominator of concentration metric
            if outcome == 'dem_race_black':
                # for fraction black in highest-risk patients,
                # denominator is the n of top k%
                total_outcome = top_k
            else:
                # for concentration in highest-risk patients,
                # denominator is the total sum of the entire holdout
                total_outcome = holdout_pred_df[outcome].sum()

            # calculate concentration metric
            frac_top_k = top_k_outcome / total_outcome

            # add column to concentration_dict (row)
            concentration_dict[OUTCOME_DICT[outcome]] = frac_top_k

            # calculate standard error (SE)
            n = len(holdout_pred_df)
            import math
            # SE = sqrt[ p * (1-p) / n]
            se = math.sqrt((frac_top_k * (1-frac_top_k))/n)

            # add SE column to concentration_dict (row)
            concentration_dict[OUTCOME_DICT[outcome] + ' SE'] = se
        all_concentration_metric.append(concentration_dict)

    # convert to pd.DataFrame for pretty formatting
    concentration_df = pd.DataFrame(all_concentration_metric)
    concentration_df = concentration_df.set_index('predictor')

    # define column order of Table 2
    column_order = []
    for outcome in outcomes:
        outcome = OUTCOME_DICT[outcome]
        column_order.append(outcome)
        column_order.append(outcome + ' SE')

    return concentration_df[column_order]


def get_best_worst_difference(df):
    """Calculate difference between best and worst for each
    outcome of interest.

    Parameters
    ----------
    df : pd.DataFrame
        Concentration metric df.

    Returns
    -------
    pd.DataFrame
        Table 2 for df.

    """
    # define dict to store 'Best-worst difference' metric for given outcome
    # (final row in Table 2)
    best_worst_dict = {
        'predictor': 'Best-worst difference'
    }

    # for each concentration of a given outcome of interest (columns),
    # calculate best - worst (same as max - min)
    for col in df.columns:
        # skip SE columns
        if 'SE' == col[-2:]:
            continue

        # calculate best - worst
        max = df[col].max()
        min = df[col].min()
        diff = max - min

        # add best - worst calculate to best_worst_dict (row)
        best_worst_dict[col] = diff

    # convert to pd.DataFrame for pretty formatting
    best_worst_row = pd.DataFrame(best_worst_dict, index=[0]).set_index('predictor')
    return best_worst_row


def build_table2(k=0.03):
    """Build Table 2 and save as CSV.

    Parameters
    ----------
    k : float
        Top k% patients in terms of predicted risk.

    Returns
    -------
    pd.DataFrame
        Table 2.
    """
    # define output dir
    git_dir = util.get_git_dir()
    OUTPUT_DIR = util.create_dir(os.path.join(git_dir, 'results'))

    # load holdout predictions generated from model
    holdout_pred_fp = os.path.join(OUTPUT_DIR, 'model_lasso_predictors.csv')
    holdout_pred_df = pd.read_csv(holdout_pred_fp)

    # calculate algorithm performance on alternative labels
    concentration_df = get_concentration_metric_df(k, holdout_pred_df)
    # calculate best - worst
    best_worst_row = get_best_worst_difference(concentration_df)

    # combine all rows to build our Table 2
    table2 = pd.concat([concentration_df, best_worst_row], sort=False)

    # save output to CSV
    filename = 'table2_concentration_metric.csv'
    output_filepath = os.path.join(OUTPUT_DIR, filename)
    print('...writing to {}'.format(output_filepath))
    table2.to_csv(output_filepath, index=True)

    return table2

if __name__ == '__main__':
    build_table2(k=0.03)
