
from pyspark.sql import functions as F

spark.read.csv('gs://outbrain-input/clicks_test.csv', header=True)\
    .createOrReplaceTempView('clicks_test')
spark.read.csv('gs://outbrain-input/clicks_train.csv', header=True)\
    .createOrReplaceTempView('clicks_train')
spark.read.csv('gs://outbrain-input/promoted_content.csv', header=True)\
    .createOrReplaceTempView('promoted_content')
spark.read.csv('gs://outbrain-input/events.csv', header=True)\
    .createOrReplaceTempView('events')
spark.read.csv('gs://outbrain-input/page_views_sample.csv', header=True)\
    .createOrReplaceTempView('page_views_sample')


test_df = spark.sql(
        """select clicks_test.*, promoted_content.document_id as ad_target, events.uuid, events.timestamp,
            CONCAT(events.uuid, '_', promoted_content.document_id) as usr_target
        from clicks_test
        left join promoted_content
            on clicks_test.ad_id = promoted_content.ad_id
        left join events
            on clicks_test.display_id = events.display_id""")

promoted_times = test_df.groupby('usr_target').agg(F.collect_list('timestamp'))

promoted_times.createOrReplaceTempView('promoted_times')

clicked = spark.sql("""select distinct promoted_times.usr_target from promoted_times INNER JOIN page_views_sample
    on promoted_times.usr_target = CONCAT(events.uuid, '_', promoted_content.document_id) """)

average_ctr = spark.sql("""select ad_id, count(clicked=1) as a, count(clicked==0) as a2, count(*) as b, count(clicked=1)/count(*) as avg_ctr from clicks_train group by ad_id""")