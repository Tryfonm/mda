import math
import re
from pathlib import Path
from typing import Dict
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

warnings.simplefilter(action='ignore')

STG = Path('./stage')
OUTPUT = Path('./output')
if not STG.exists():
    STG.mkdir(parents=True)
    print("Directory created:", STG)
else:
    print("Directory already exists:", STG)

if not OUTPUT.exists():
    OUTPUT.mkdir(parents=True)
    print("Directory created:", OUTPUT)
else:
    print("Directory already exists:", OUTPUT)


MONTHS = [
    'Jan',
    'Feb',
    'March',
    'April',
    'May',
    'June',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec'
]

LOCATIONS = {
    'naamsestraat_35': (50.87714960158344, 4.700721880135347),
    'naamsestraat_57': (50.87649863917443, 4.700715782211919),
    'naamsestraat_62': (50.87584845432454, 4.700202229442342),
    'naamsestraat_76': (50.87526873692704, 4.700112669705018),
    'calvariekapel': (50.87451212733984, 4.6999140919018085),
    'paarkstraat_2': (50.87412447017892, 4.7000178233016525),
    'naamsestraat_81': (50.87383596405986, 4.700120392891415),
    'kiosk_stadspark': (50.87533314504303, 4.701502735494995),
    'vrijthof': (50.8789238922419, 4.701195448073587)
}


def get_dict_of_dfs(location):
    """Given the location as an input argument, a dictionary with month-keys is returned

    Args:
        location (_type_): _description_
    """
    dataset_dir = Path('dataset')

    raw_dataset = {}
    for month in tqdm(MONTHS, desc=f'Processing'):
        current_dir = Path.joinpath(dataset_dir, month)
        concatenated_data = pd.DataFrame()

        raw_dataset[month] = {}
        for file_path in current_dir.iterdir():

            if file_path.suffix == ".csv":
                if re.search(location, str(file_path), re.IGNORECASE):
                    # print(file_path)
                    df = pd.read_csv(file_path, delimiter=';')
                    concatenated_data = pd.concat(
                        [concatenated_data, df], ignore_index=True
                    )
        raw_dataset[month] = concatenated_data

    return raw_dataset


def pivot_into_single_df(input_dict: Dict[str, pd.DataFrame]) ->pd.DataFrame:
    """Helper for loading data from multiple dfs into a single one

    Args:
        input_dict (Dict[str, pd.DataFrame]):

    Returns:
        _type_: _description_
    """
    concatenated_df = pd.concat(
        input_dict.values(), ignore_index=True
    )

    return concatenated_df


def preprocess_noise_df(
    raw_df: pd.DataFrame,
    impute_na_method: str = 'linear',
    resample_arg: str = '30T',
):
    """Preprocessing helper for noise dataset
    
    Args:
        raw_df (pd.DataFrame):
        impute_na_method (str, optional): Defaults to 'linear'.
        resample_arg (str, optional): Defaults to '30T'.

    """
    raw_df['result_timestamp'] = pd.to_datetime(
        raw_df['result_timestamp'], format='%d/%m/%Y %H:%M:%S.%f'
    )
    raw_df.drop(
        columns=[
            '#object_id',
            'description',
            'lamax_unit',
            'laeq_unit',
            'lceq_unit',
            'lcpeak_unit'
        ], inplace=True
    )

    if impute_na_method:
        raw_df['lamax'] = raw_df['lamax'].interpolate(method=impute_na_method)

    if resample_arg:
        raw_df.set_index('result_timestamp', inplace=True)
        return raw_df.resample(resample_arg).mean()

    return raw_df


def preprocess_meteo_df(
    raw_df: pd.DataFrame,
    impute_na_method: str = 'linear',
    resample_arg: str = '30T',
):
    """Preprocessing helper for meteo dataset

    Args:
        raw_df (pd.DataFrame): _description_
        impute_na_method (str, optional): _description_. Defaults to 'linear'.
        resample_arg (str, optional): _description_. Defaults to '30T'.

    """
    raw_df['DATEUTC'] = raw_df['DATEUTC'].str.replace('2020-', '2022-')
    raw_df = raw_df[~(raw_df['DATEUTC'].str.contains('2022-02-29'))]
    raw_df['DATEUTC']
    raw_df['DATEUTC'] = pd.to_datetime(
        raw_df['DATEUTC'], format='mixed'  # '%Y-%m-%d %H:%M:%S'
    )
    raw_df.drop(
        columns=[
            'ID',
            'Date',
            'Year',
            'Month',
            'Day',
            'Hour',
            'Minute'
        ], inplace=True
    )

    if resample_arg:
        raw_df.set_index('DATEUTC', inplace=True)
        raw_df = raw_df.resample(resample_arg).mean()

    if impute_na_method:
        raw_df = raw_df.interpolate(method=impute_na_method)

    return raw_df


def plotly_wrapper(df: pd.DataFrame, feature: str = 'laeq', location: str = 'LAEQ'):
    """
    Args:
        df (pd.DataFrame): 
        feature (str, optional): Defaults to 'laeq'.
        location (str, optional): Defaults to 'LAEQ'.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[feature], mode='lines', name=feature)
    )

    fig.update_layout(
        title=f'Time Series: {feature} - {location}',
        xaxis_title='Timestamp',
        yaxis_title='LAEQ'
    )

    fig.show()


def heatmap_wrapper(may_corr: pd.DataFrame, june_corr: pd.DataFrame, july_corr: pd.DataFrame):
    """
    Args:
        may_corr (pd.DataFrame): _description_
        june_corr (pd.DataFrame): _description_
        july_corr (pd.DataFrame): _description_
    """
    fig = plt.figure(figsize=(15, 5))
    grid = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(grid[0, 0])
    sns.heatmap(may_corr, annot=True, cmap='coolwarm', annot_kws={"size": 5})
    ax1.set_title("May")

    ax2 = fig.add_subplot(grid[0, 1])
    sns.heatmap(june_corr, annot=True, cmap='coolwarm', annot_kws={"size": 5})
    ax2.set_title("June")

    ax3 = fig.add_subplot(grid[0, 2])
    sns.heatmap(july_corr, annot=True, cmap='coolwarm', annot_kws={"size": 5})
    ax3.set_title("July")

    plt.tight_layout()

    # Display the plot
    plt.show()


def test_granger_causality(series1, series2, maxlag=4):
    """Given two pd.Series the function displays statistics related to granger-causality test
    Args:
        series1 (_type_): 
        series2 (_type_): 
        maxlag (int, optional): Defaults to 4.
    """
    # Combine the series into a DataFrame
    data = pd.DataFrame({'series1': series1, 'series2': series2})

    # Drop rows with NaN values
    data = data.dropna()

    # Run the Granger causality test
    results = grangercausalitytests(data, maxlag=maxlag)

    # Print the results
    for lag in results.keys():
        print(f"Lag {lag}:")
        print("Test statistic:", results[lag][0]['ssr_ftest'][0])
        print("P-value:", results[lag][0]['ssr_ftest'][1])
        print()


def calculate_distance(location_1, location_2):
    earth_radius = 6371
    location_1 = LOCATIONS[location_1]
    location_2 = LOCATIONS[location_2]

    lat1_rad = math.radians(location_1[0])
    lon1_rad = math.radians(location_1[1])
    lat2_rad = math.radians(location_2[0])
    lon2_rad = math.radians(location_2[1])

    # Haversine formula
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = math.sin(delta_lat/2) ** 2 + math.cos(lat1_rad) * \
        math.cos(lat2_rad) * math.sin(delta_lon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = earth_radius * c

    return distance


def create_sequences(input_data, target_column, sequence_length):
    """Helper function for creating Time series torch.Datasets

    Args:
        input_data (_type_): 
        target_column (_type_):
        sequence_length (_type_):

    """
    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - sequence_length)):
        sequence = input_data[i:i+sequence_length]

        label_position = i + sequence_length
        label = input_data.iloc[label_position][target_column]

        sequences.append((sequence, label))

    return sequences


if __name__ == '__main__':
    pass
