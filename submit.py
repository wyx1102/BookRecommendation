from googleapiclient import discovery
from googleapiclient import errors
from datetime import datetime
from oauth2client.client import GoogleCredentials
import logging
import os

from google.cloud import storage

def delete_blob(bucket_name, blob_name):
        """Deletes a blob from the bucket."""
    
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()

        print("Blob {} deleted.".format(blob_name))

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        print(blob.name)
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
def check(bucket_name, dirname):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        
        if((blob.name).startswith(dirname)):
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob)
            blob.delete()
def main():
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    JOBNAME= "wals_"+ dt_string
    OUTDIR="gs://{}/wals/model_trained".format(os.environ["BUCKET"])
    num_epochs = str(10)
    NITEMS = 367982
    NUSERS = 603668
    
    check(os.environ["BUCKET"], OUTDIR)

    training_inputs = {
        'scaleTier': 'BASIC_GPU',
        'packageUris': ['gs://amazonbookrecommendation/code/wals_ml_engine-0.1.tar.gz'],
        'pythonModule': 'trainer.task',
        'args': ['--output_dir', OUTDIR, '--input_path', 'gs://{}/wals/data'.format(os.environ["BUCKET"]), '--num_epochs', num_epochs, '--nitems', str(NITEMS), '--nusers', str(NUSERS)],
        'region': 'us-east1',
        'jobDir': 'gs://{}/wals/model_trained'.format(os.environ["BUCKET"]),
        'runtimeVersion': os.environ["TFVERSION"],
        'pythonVersion': '3.7',
        'scheduling': {'maxRunningTime': '7200s'},
    }
   
    job_spec = {'jobId': JOBNAME, 'trainingInput': training_inputs}
   
    project_id = 'projects/{}'.format(os.environ["PROJECT"])
    project_name = os.environ["PROJECT_NAME"]
    cloudml = discovery.build('ml', 'v1')
    request = cloudml.projects().jobs().create(body=job_spec,
                  parent=project_id)

    try:
        response = request.execute()
        print("running job : {}".format(JOBNAME))
        print("job state : {}".format(response['state']))
    except errors.HttpError as err:
        logging.error('There was an error creating the training job.'
                      ' Check the details:')
        logging.error(err._get_reason())
        

    