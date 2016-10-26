    
from pyspark import SparkContext
from operator import add

TEST_CSV = 's3n://michaldul/kaggle-outbrain/clicks_test.csv'
TRAIN_CSV = 's3n://michaldul/kaggle-outbrain/clicks_train.csv'
OUTPUT_CSV = 's3n://michaldul/kaggle-outbrain/output.csv'

if __name__ == "__main__":
    sc = SparkContext(appName="Outbrain BTB")

    clicks_train = sc.textFile(TRAIN_CSV) \
        .filter(lambda line: line[0] != 'd') \
        .map(lambda line: line.split(',')) \
        .map(lambda row: (int(row[1]), int(row[2])))

    ad_clicks = clicks_train.reduceByKey(add)
    ad_shows = clicks_train.map(lambda ad: (ad[0], 1)).reduceByKey(add)
    ratios = ad_clicks.leftOuterJoin(ad_shows) \
        .map(lambda row: (row[0], row[1][0] / ((row[1][1] or 1) + 10)))


    def sorted_by_prob(l):
        l.sort(key=lambda pair: pair[1] or 0, reverse=True)
        return ' '.join([str(pair[0]) for pair in l])


    clicks_test = sc.textFile(TEST_CSV) \
        .filter(lambda line: line[0] != 'd') \
        .map(lambda line: line.split(',')) \
        .map(lambda row: (int(row[1]), int(row[0]))) \
        .leftOuterJoin(ratios) \
        .map(lambda row: (int(row[1][0]), (int(row[0]), row[1][1]))) \
        .aggregateByKey([], lambda t, r: t + [r], lambda a, b: a + b) \
        .map(lambda row: "{0},{1}".format(row[0], sorted_by_prob(row[1]))) \
        .saveAsTextFile(OUTPUT_CSV)

