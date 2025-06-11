"""
Build Table 1: summary statistics by race.
"""
import pandas as pd
import os
import model.util as util


def get_table1(df, race, total_n):
    """Build Table 1.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of data dataframe by race
    race : str
        Patient race data subset, used as column name for Table 1.
    total_n : int
        Total number of observations in entire dataframe.

    Returns
    -------
    pd.DataFrame
        Table 1 for df.
    """
    print('...building table1 for {}'.format(race))
    df = df.copy()
    table1_list = []  # list of tuples

    n = len(df) * 1.0
    table1_list.append(('n (patient-years)', '{:,}'.format(n)))

    # Demographics
    print('....adding demographics')
    table1_list.append(('Demographics', '---'))
    table1_list.append(('Age 18-24','{:.2f}'.format(df['dem_age_band_18-24_tm1'].sum() / n)))
    table1_list.append(('Age 25-34', '{:.2f}'.format(df['dem_age_band_25-34_tm1'].sum() / n)))
    table1_list.append(('Age 35-44', '{:.2f}'.format(df['dem_age_band_35-44_tm1'].sum() / n)))
    table1_list.append(('Age 45-54', '{:.2f}'.format(df['dem_age_band_45-54_tm1'].sum() / n)))
    table1_list.append(('Age 55-64', '{:.2f}'.format(df['dem_age_band_55-64_tm1'].sum() / n)))
    table1_list.append(('Age 65-74', '{:.2f}'.format(df['dem_age_band_65-74_tm1'].sum() / n)))
    table1_list.append(('Age 75+', '{:.2f}'.format(df['dem_age_band_75+_tm1'].sum() / n)))
    table1_list.append(('Female', '{:.2f}'.format(df['dem_female'].sum() / n)))

    # Care management program
    print('....adding care management program')
    table1_list.append(('Care management program', '---'))
    df['risk_score_t_percentile'] = util.convert_to_percentile(df, 'risk_score_t').astype(int)
    table1_list.append(('Algorithm score (percentile)', '{:.0f}'.format(df['risk_score_t_percentile'].mean())))
    table1_list.append(('Race composition of program (%)', '{:.1f}'.format(n / total_n * 100)))

    # Care utilization
    print('....adding care utilization')
    table1_list.append(('Care utilization', '---'))
    table1_list.append(('Actual cost', '${:,.0f}'.format(df['cost_t'].mean())))

    # Mean biomarker values
    print('....adding mean biomarkers')
    table1_list.append(('Mean biomarkers', '---'))
    table1_list.append(('HbA1c', '{:.1f}'.format(df['ghba1c_mean_t'].mean())))
    table1_list.append(('Systolic BP', '{:.1f}'.format(df['bps_mean_t'].mean())))
    table1_list.append(('Creatinine', '{:.1f}'.format(df['cre_mean_t'].mean())))
    table1_list.append(('Hematocrit', '{:.1f}'.format(df['hct_mean_t'].mean())))
    table1_list.append(('LDL', '{:.1f}'.format(df['ldl_mean_t'].mean())))

    # Active chronic illnesses (comorbidities)
    print('....adding active chronic illnesses (comorbidities)')
    table1_list.append(('Active chronic illnesses (comorbidities)', '---'))
    table1_list.append(('Total number of active illnesses', '{:.2f}'.format(df['gagne_sum_t'].mean())))
    table1_list.append(('Hypertension',
        '{:.2f}'.format(df['hypertension_elixhauser_tm1'].mean())))
    table1_list.append(('Diabetes, uncomplicated', '{:.2f}'.format(df['uncompdiabetes_elixhauser_tm1'].mean())))
    table1_list.append(('Arrythmia',
        '{:.2f}'.format(df['arrhythmia_elixhauser_tm1'].mean())))
    table1_list.append(('Hypothyroid', '{:.2f}'.format(df['hypothyroid_elixhauser_tm1'].mean())))
    table1_list.append(('Obesity', '{:.2f}'.format(df['obesity_elixhauser_tm1'].mean())))
    table1_list.append(('Pulmonary disease', '{:.2f}'.format(df['pulmonarydz_romano_tm1'].mean())))
    table1_list.append(('Cancer', '{:.2f}'.format(df['tumor_romano_tm1'].mean())))
    table1_list.append(('Depression', '{:.2f}'.format(df['depression_elixhauser_tm1'].mean())))
    table1_list.append(('Anemia', '{:.2f}'.format(df['anemia_elixhauser_tm1'].mean())))
    table1_list.append(('Arthritis', '{:.2f}'.format(df['arthritis_elixhauser_tm1'].mean())))
    table1_list.append(('Renal failure', '{:.2f}'.format(df['renal_elixhauser_tm1'].mean())))
    table1_list.append(('Electrolyte disorder', '{:.2f}'.format(df['electrolytes_elixhauser_tm1'].mean())))
    table1_list.append(('Heart failure', '{:.2f}'.format(df['chf_romano_tm1'].mean())))
    table1_list.append(('Psychosis', '{:.2f}'.format(df['psychosis_elixhauser_tm1'].mean())))
    table1_list.append(('Valvular disease', '{:.2f}'.format(df['valvulardz_elixhauser_tm1'].mean())))
    table1_list.append(('Stroke', '{:.2f}'.format(df['hemiplegia_romano_tm1'].mean())))
    table1_list.append(('Peripheral vascular disease', '{:.2f}'.format(df['pvd_elixhauser_tm1'].mean())))
    table1_list.append(('Diabetes, complicated', '{:.2f}'.format(df['compdiabetes_elixhauser_tm1'].mean())))
    table1_list.append(('Heart attack', '{:.2f}'.format(df['myocardialinfarct_romano_tm1'].mean())))
    table1_list.append(('Liver disease', '{:.2f}'.format(df['liver_elixhauser_tm1'].mean())))

    table1_df = pd.DataFrame(table1_list, columns=['Descriptive stats', race])

    return table1_df


def build_table1():
    """Build Table 1 and save as CSV."""
    # define output dir
    git_dir = util.get_git_dir()
    OUTPUT_DIR = util.create_dir(os.path.join(git_dir, 'results'))

    # define filepath
    git_dir = util.get_git_dir()
    data_fp = os.path.join(git_dir, 'data', 'data_new.csv')

    # load df
    data_df = pd.read_csv(data_fp)

    # calculate total N for entire df
    total_n = len(data_df) * 1.0

    # split by white, black patients
    white_df = data_df[data_df['race'] == 'white']
    black_df = data_df[data_df['race'] == 'black']

    white_table1 = get_table1(white_df, 'White', total_n)
    black_table1 = get_table1(black_df, 'Black', total_n)

    # merge white table 1 and black table 1
    table1 = white_table1.merge(black_table1)

    # save output to CSV
    filename = 'table1.csv'
    output_filepath = os.path.join(OUTPUT_DIR, filename)
    print('...writing to {}'.format(output_filepath))
    table1.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    build_table1()
