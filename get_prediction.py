from google.cloud import bigquery
# import prediction file into bigquery and get prediction image urls
def main(user = 'AVC8ZAFPYOHZL'):
    client = bigquery.Client(project='bookrecommendation-267223')
     # download prediction data and import into bigquery
    dataset_id = 'data'
    dataset_ref = client.dataset(dataset_id)
    job_config = bigquery.LoadJobConfig()

    job_config.schema = [
        bigquery.SchemaField("userId", "STRING"),
        bigquery.SchemaField("asin1", "STRING"),
        bigquery.SchemaField("asin2", "STRING"),
        bigquery.SchemaField("asin3", "STRING")
    ]
    job_config.source_format = bigquery.SourceFormat.CSV
    uri = "gs://amazonbookrecommendation/wals/model_trained/batch_pred.csv"
    client.delete_table('bookrecommendation-267223.data.prediction', not_found_ok=True) 
    load_job = client.load_table_from_uri(
        uri, dataset_ref.table("prediction"), job_config=job_config
    )  # API requests
    print("Starting job {}".format(load_job.job_id))
    load_job.result()  # Waits for table load to complete.
    print("Job finished.")
    destination_table = client.get_table(dataset_ref.table("prediction"))
    print("Loaded {} rows.".format(destination_table.num_rows))


    # query the prediction table
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user", "STRING", user)
        ]
    )
    sql = """
        select 
            asin, imUrl 
        from 
            bookrecommendation-267223.data.meta, 
            (select * from bookrecommendation-267223.data.prediction where userId = @user) as books
        where 
            asin = books.asin1 or asin = books.asin2 or asin = books.asin3
    """
    query_job = client.query(sql, job_config = job_config)
    df = query_job.to_dataframe()
    urls =  df['imUrl'].tolist()
    return urls
