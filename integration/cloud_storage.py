 import boto3
 from botocore.exceptions import NoCredentialsError
 import logging

 # Configure logging
 logging.basicConfig(filename='../logs/cloud_storage.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 def upload_to_s3(file_name, bucket, object_name=None):
     s3_client = boto3.client('s3')
     try:
         s3_client.upload_file(file_name, bucket, object_name or file_name)
         logging.info(f"File {file_name} uploaded to {bucket} as {object_name or file_name}")
     except FileNotFoundError:
         logging.error(f"The file {file_name} was not found.")
     except NoCredentialsError:
         logging.error("Credentials not available.")
