from ucimlrepo import fetch_ucirepo # database import from package ucimlrepo

def database():
    # fetch dataset
    wine = fetch_ucirepo(id=109)

    # data (as pandas dataframes)
    X = wine.data.features
    y = wine.data.targets

    # metadata
    print(wine.metadata)

    # variable information
    print(wine.variables)