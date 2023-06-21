import pandas as pd
import yaml
import sys
import random
import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)


params = yaml.safe_load(open("params.yaml"))["prepare"]

city_vectoriser_path = sys.argv[1]
team_vectoriser_path = sys.argv[2]
data_path = sys.argv[3]

with open(city_vectoriser_path, "rb") as fd:
    city_vectoriser = pickle.load(fd)

with open(city_vectoriser_path, "rb") as fd:
    team_vectoriser = pickle.load(fd)


os.makedirs(params['to_save_path'], exist_ok=True)



def get_season(date_time):
    match date_time.month_name():
        case 'December' | 'January' | 'February':
            return 'Winter'
        case 'March' | 'April' | 'May':
            return 'Spring'
        case 'June' | 'August' | 'July':
            return 'Summer'
        case 'September' | 'October' | 'November':
            return 'Autumn'
        

def get_day_part(date_time):
    if date_time.hour > 0 and date_time.hour <= 12:
        return 'Morning'
    elif date_time.hour > 12 and date_time.hour <= 18:
        return 'Day'
    else:
        return 'Evening'
    

def to_std(df, columns):
    res = Pipeline([
        ('StandardScaler', StandardScaler()),
        ('MinMaxScaler', MinMaxScaler())
    ]).fit_transform(df[columns])

    for i, col_name in enumerate(columns):
        df[col_name] = res[:, i]
    
    return df

        


data_farme = pd.read_csv(data_path)
data_farme['MATCH_DATE'] = pd.to_datetime(data_farme['MATCH_DATE'])

data_farme.drop(['Unnamed: 0', 'VIEWER', 'LEAGUE', 'FINISHED', 'SEASON', 'HOME_TEAM_NAME', 'HOME_TEAM_ID', 'HOME_ICON', 'AWAY_TEAM_NAME', 'AWAY_TEAM_ID', 'AWAY_ICON'], axis=1, inplace=True)

data_farme['AWAY_TEAM'] = data_farme['AWAY_TEAM'].fillna("UNIKNOW")

df_mirror = data_farme.copy()

data_farme.rename(columns={
    'HOME_TEAM': 'TEAM', 
    'AWAY_TEAM': 'RIVAl',
    'WIN_HOME': 'IS_WIN',
    'GOALS_HOME': 'TEAM_GOALS_NUM',
    'GOALS_AWAY': 'RIVAl_GOALS_NUM'}, inplace=True)
data_farme.drop(['WIN_AWAY'], axis=1, inplace=True)
data_farme['IS_HOME_GAME'] = True

df_mirror.rename(columns={
    'HOME_TEAM': 'RIVAl', 
    'AWAY_TEAM': 'TEAM',
    'WIN_AWAY': 'IS_WIN',
    'GOALS_AWAY': 'TEAM_GOALS_NUM',
    'GOALS_HOME': 'RIVAl_GOALS_NUM'}, inplace=True)
df_mirror.drop(['WIN_HOME'], axis=1, inplace=True)
df_mirror['IS_HOME_GAME'] = False

df = pd.concat([data_farme, df_mirror])

df['YEAR_SEASON'] = df['MATCH_DATE'].apply(get_season)
df['DAY_PART'] = df['MATCH_DATE'].apply(get_day_part)
df['DAY_OF_WEEK'] = df['MATCH_DATE'].dt.day_of_week
df.drop('MATCH_DATE', axis=1, inplace=True)

#df['IS_WIN'] = df['IS_WIN'].astype(bool)


city_vect = city_vectoriser.transform(df['LOCATION'])
city_vect_df = pd.DataFrame(city_vect.toarray(), columns=["city_" + str(i) for i in range(city_vect.shape[1])])
df = df.join(city_vect_df)

team_name_vect = team_vectoriser.transform(df['TEAM'])
team_name_vect_df = pd.DataFrame(team_name_vect.toarray(), columns=["team_" + str(i) for i in range(team_name_vect.shape[1])])
df = df.join(team_name_vect_df)

rival_name_vect = team_vectoriser.transform(df['RIVAl'])
rival_name_vect_df = pd.DataFrame(rival_name_vect.toarray(), columns=["rival_" + str(i) for i in range(rival_name_vect.shape[1])])
df = df.join(rival_name_vect_df)

df = pd.get_dummies(df, columns=['YEAR_SEASON', 'DAY_PART', 'DAY_OF_WEEK'], drop_first=True)
df['IS_WIN_NUM'] = df['IS_WIN']



#########
df.drop(['LEAGUE_NAME', 'IS_WIN', 'LOCATION', 'TEAM', 'RIVAl', 'MATCHDAY'], axis=1, inplace=True)
#########

test_df, train_df = train_test_split(df, train_size=params['split'], shuffle=True, random_state=params['seed'])

std_columns = df.loc[:, df.columns !='IS_WIN_NUM'].columns

train_df = to_std(train_df, std_columns)
test_df = to_std(test_df, std_columns)


train_df.to_csv(os.path.join("data", "prepared", "train.csv"), index=False)
test_df.to_csv(os.path.join("data", "prepared", "test.csv"), index=False)
