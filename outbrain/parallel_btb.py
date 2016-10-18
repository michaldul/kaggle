import dask.dataframe as dd

reg = 10  # trying anokas idea of regularization


def generate_unrolled_output():
    train = dd.read_csv("./data/clicks_train.csv")
    cnt = train[train.clicked == 1].ad_id.value_counts()
    cntall = train.ad_id.value_counts()
    prop = (cnt / (cntall + 10)).fillna(0).compute()

    test = dd.read_csv("./data/clicks_test.csv")
    test = test.assign(prop=test['ad_id'].map(prop))

    test.to_csv("./unrolled.csv", index=False)

if __name__ == '__main__':
    generate_unrolled_output()