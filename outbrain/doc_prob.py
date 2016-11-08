

import pandas as pd
import numpy as np
from ml_metrics import mapk


def load_train():
    events = pd.read_csv("./input/events.csv", usecols=['display_id', 'document_id'])
    train = pd.read_csv("./input/clicks_train.csv")
    return pd.merge(train, events, on='display_id')


def split_train(df):
    ids = df.display_id.unique()
    ids = np.random.choice(ids, size=len(ids) // 10, replace=False)

    valid = df[df.display_id.isin(ids)]
    train = df[~df.display_id.isin(ids)]

    return train, valid


def predict(train, valid):
    merged = pd.merge(valid[['document_id', 'ad_id']].drop_duplicates(), train, on=('document_id', 'ad_id'))
    clicks = merged[merged.clicked == 1].groupby(('document_id', 'ad_id')).display_id.count()
    shows = merged.groupby(('document_id', 'ad_id')).display_id.count()

    return clicks, shows


def get_get_prob(cnt, cntall, doc):
    def get_prob(k):
        if (doc, k) not in cnt:
            return 0
        return cnt[doc, k] / (float(cntall[doc, k]) + 1)

    return get_prob

if __name__ == '__main__':
    df = load_train()
    train, valid = split_train(df)

    clicks, shows = predict(train, valid)

    y = valid[valid.clicked == 1].ad_id.values
    y = [[_] for _ in y]

    p = valid.groupby(['display_id', 'document_id']).ad_id.apply(list)
    p = [sorted(x, key=get_get_prob(clicks, shows, doc), reverse=True) for (display, doc), x in p.iteritems()]

    print(mapk(y, p, k=12))
