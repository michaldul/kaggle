# gcloud dataproc clusters create cluster-1 --zone europe-west1-b --master-machine-type n1-standard-1 --master-boot-disk-size 50 --num-workers 2 --worker-machine-type n1-highmem-2 --worker-boot-disk-size 50 --project kaggle-outbrain-149201
# gcloud dataproc clusters create cluster-2 --zone europe-west1-b --master-machine-type n1-standard-1 --master-boot-disk-size 50 --num-workers 5 --worker-machine-type n1-highmem-4 --worker-boot-disk-size 50 --project kaggle-outbrain-149201


from pyspark.sql import functions as F

spark.read.csv('gs://outbrain-input/clicks_test.csv', header=True)\
    .createOrReplaceTempView('clicks_test')
spark.read.csv('gs://outbrain-input/clicks_train.csv', header=True)\
    .createOrReplaceTempView('clicks_train')
spark.read.csv('gs://outbrain-input/promoted_content.csv', header=True)\
    .createOrReplaceTempView('promoted_content')
spark.read.csv('gs://outbrain-input/events.csv', header=True)\
    .createOrReplaceTempView('events')
spark.read.csv('gs://outbrain-input/page_views.csv', header=True)\
    .createOrReplaceTempView('page_views')


test_df = spark.sql(
        """select clicks_test.*, promoted_content.document_id as ad_target, events.uuid, events.timestamp,
            CONCAT(events.uuid, '_', promoted_content.document_id) as usr_target
        from clicks_test
        left join promoted_content
            on clicks_test.ad_id = promoted_content.ad_id
        left join events
            on clicks_test.display_id = events.display_id""")

test_df.groupby('usr_target').agg(F.collect_list('timestamp')).createOrReplaceTempView('promoted_times')

spark.sql("""select distinct promoted_times.usr_target
    from promoted_times INNER JOIN page_views
    on promoted_times.usr_target = CONCAT(page_views.uuid, '_', page_views.document_id) """)\
    .createOrReplaceTempView('clicked')


spark.sql("""SELECT ad_id, SUM(clicked)/(COUNT(*) + 10) AS avg_ctr, COUNT(*) AS shows
        FROM clicks_train
        GROUP BY ad_id""").createOrReplaceTempView('average_ctr')


test_prob = spark.sql("""select clicks_test.display_id, clicks_test.ad_id,
            CASE WHEN clicked.usr_target IS NULL THEN
              CASE WHEN average_ctr.avg_ctr = 0 THEN
                -shows
                ELSE COALESCE (average_ctr.avg_ctr, (SELECT SUM(clicked)/COUNT(*) FROM clicks_train)) END
              ELSE 1
            END AS prob
            from clicks_test
            join events on clicks_test.display_id = events.display_id
            left join clicked on clicked.usr_target = CONCAT(events.uuid, '_', events.document_id)
            left join average_ctr on clicks_test.ad_id = average_ctr.ad_id""")


test_prob.rdd \
    .map(lambda row: (row.display_id, (row.ad_id, row.prob)))\
    .groupByKey()\
    .map(lambda (k, v): (k, ' '.join([p[0] for p in sorted(v, key=lambda p: p[1], reverse=True)])))\
    .map(lambda (k, v): k + ',' + v).saveAsTextFile('gs://outbrain-input/sub4.csv')