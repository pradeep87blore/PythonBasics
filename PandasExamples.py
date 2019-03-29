# In this file, we will try to make use of pandas dataframes and series

import pandas as pd
import numpy as np

def read_population_data():
    df = pd.read_csv('../Data/populationbycountry19802010millions.csv')
    print(df.head)


def analyse_population_data():
    read_population_data()


if __name__ == "__main__":
    analyse_population_data()




