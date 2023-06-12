import pandas as pd
import yaml
import sys
import random
import os
import pickle

from sklearn.feature_extraction.text import HashingVectorizer



if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython create_vectorisers.py data-file\n")
    sys.exit(1)


params = yaml.safe_load(open("params.yaml"))["create_vectorisers"]
random.seed(params['seed'])

df = pd.read_csv(sys.argv[1])

os.makedirs(sys.argv[2], exist_ok=True)

home_team_id_list = df['HOME_TEAM_ID'].value_counts().keys()
city_list = df['LOCATION'].value_counts().keys()

team_id_vectoriser = HashingVectorizer(n_features=home_team_id_list.shape[0], binary=True)
team_id_vectoriser.fit(home_team_id_list)

city_vectoriser = HashingVectorizer(n_features=params['city_vct_num'], binary=True)
city_vectoriser.fit(city_list)


with open(
    os.path.join(sys.argv[2], params['team_vct_name']), "wb"
    ) as fd:
    pickle.dump(team_id_vectoriser, fd)


with open(
    os.path.join(sys.argv[2], params['city_vct_name']), "wb"
    ) as fd:
    pickle.dump(city_vectoriser, fd)
