import pandas as pd

map_task = {}
map_prc = {}
ls_fnc = []


def run_validation(path):
    df = pd.read_csv(path, parse_dates=['Date'])

    result = pd.DataFrame(columns=['User', 'Process', 'Unit', 'Time'])
    for user in df['User'].unique():
        data_user = df[df['User'] == user]
        data_user.sort_values(by=['Date'], inplace=True)

        ls_task = data_user['Task'].unique().tolist()

        for row in data_user.iterrows():



    return df







