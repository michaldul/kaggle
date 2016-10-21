
from operator import add

TEST_CSV = 'c:\\dev\\kaggle\\outbrain\\data\\clicks_test.csv'
TRAIN_CSV = 'c:\\dev\\kaggle\\outbrain\\data\\clicks_train.csv'
OUTPUT_CSV = 'c:\\dev\\kaggle\\outbrain\\data\\spark_output'

clicks_train = sc.textFile(TRAIN_CSV) \
    .filter(lambda line: line[0] != 'd') \
    .map(lambda line: line.split(',')) \
    .map(lambda row: (int(row[1]), int(row[2])))

ad_clicks = clicks_train.reduceByKey(add)
ad_shows = clicks_train.map(lambda ad: (ad[0], 1)).reduceByKey(add)
ratios = ad_clicks.leftOuterJoin(ad_shows) \
    .map(lambda row: (row[0], row[1][0]/(row[1][1] or 1)))

def sorted_by_prob(l):
    l.sort(key=lambda pair: pair[1], reverse=True)
    return ','.join([str(pair[0]) for pair in l])

clicks_test = sc.textFile(TEST_CSV) \
    .filter(lambda line: line[0] != 'd') \
    .map(lambda line: line.split(',')) \
    .map(lambda row: (int(row[1]), int(row[0]))) \
    .join(ratios) \
    .map(lambda row: (int(row[1][0]), (int(row[0]), row[1][1]))) \
    .aggregateByKey([], lambda t, r: t + [r], lambda a, b: a + b) \
    .map(lambda row: "{0};{1}".format(row[0], sorted_by_prob(row[1]))) \
    .saveAsTextFile(OUTPUT_CSV)
