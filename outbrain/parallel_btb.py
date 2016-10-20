
# 2min 45s/3min 34s (with compute)
def generate_unrolled_output():
    import dask.dataframe as dd

    reg = 10  # trying anokas idea of regularization

    train = dd.read_csv("./data/clicks_train.csv")
    cnt = train[train.clicked == 1].ad_id.value_counts()
    cntall = train.ad_id.value_counts()
    prop = (cnt / (cntall + 10)).fillna(0).compute()

    test = dd.read_csv("./data/clicks_test.csv")
    test = test.assign(prop=test['ad_id'].map(prop))

    test.compute().to_csv("./unrolled.csv", index=False)


def ads_ordered(df):
    return ' '.join(df.sort_values(by='prop', ascending=False)['ad_id'].astype(str))


# 52min 33s
def group_results():
    import dask.dataframe as dd

    dd.read_csv("./unrolled.csv").groupby('display_id')\
        .apply(ads_ordered, meta=('ad_id', 'str'))\
        .to_csv('./results.csv')

# 49min 44s
def group_results_pandas():
    import pandas as pd

    pd.read_csv("./unrolled.csv").groupby('display_id')\
        .apply(ads_ordered)\
        .to_csv('./results.csv')


# PySpark approach - 3min 32s
def sorted_by_prob(l):
    l.sort(key=lambda pair: pair[1], reverse=True)
    return ','.join([str(pair[0]) for pair in l])

sc.textFile('c:\\dev\\kaggle\\outbrain\\unrolled_no_header.csv') \
    .map(lambda row: row.split(',')) \
    .map(lambda row: (int(row[0]), (int(row[1]), float(row[2] or 0)))) \
    .aggregateByKey([], lambda t, r: t + [r], lambda a, b: a + b) \
    .map(lambda row: "{0};{1}".format(row[0], sorted_by_prob(row[1]))) \
    .saveAsTextFile('output_spark.csv')



if __name__ == '__main__':
    group_results_pandas()
