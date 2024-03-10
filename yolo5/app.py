import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
import json
import requests

images_bucket = os.environ['BUCKET_NAME']
queue_name = os.environ['SQS_QUEUE_NAME']
REGION_NAME = os.environ['REGION_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']
s3 = boto3.client('s3')
s3_client = boto3.client('s3', region_name=REGION_NAME)
sqs_client = boto3.client('sqs', region_name=REGION_NAME)
dynamodb_client = boto3.client('dynamodb', region_name=REGION_NAME)


def consume():
    while True:
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        if 'Messages' in response:
            message = response['Messages'][0]['Body']
            receipt_handle = response['Messages'][0]['ReceiptHandle']

            # Use the ReceiptHandle as a prediction UUID
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'prediction: {prediction_id}. start processing')

            message_data = json.loads(message)

            # Receives a URL parameter representing the image to download from S3
            s3_photo_path = message_data.get('s3_photo_path')  # TODO extract from `message`
            chat_id = message_data.get('chat_id')  # TODO extract from `message`
            local_dir = 'photos/'  # str of dir to save to
            os.makedirs(local_dir, exist_ok=True)  # make sure the dir exists
            filename = s3_photo_path.split('/')[-1]
            original_img_path = filename
            s3.download_file(images_bucket, s3_photo_path, original_img_path)  # TODO download img_name from S3,
            # store the
            # local image path in original_img_path

            logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

            # This is the path for the predicted image with labels The predicted image typically includes bounding
            # boxes drawn around the detected objects, along with class labels and possibly confidence scores.
            predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

            # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original
            #  image).
            predicted_img_name = f'predicted_{filename}'  # assign the new name
            os.rename(f'/usr/src/app/static/data/{prediction_id}/{filename}',
                      f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}')  # rename the file before upload
            s3_path_to_upload_to = '/'.join(
                s3_photo_path.split('/')[:-1]) + f'/{predicted_img_name}'  # assign the path on s3 as str
            file_to_upload = f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}'  # assign the path
            # locally as str
            s3.upload_file(file_to_upload, images_bucket,
                           s3_path_to_upload_to)  # upload the file to same path with new name s3
            os.rename(f'/usr/src/app/static/data/{prediction_id}/{predicted_img_name}',
                      f'/usr/src/app/static/data/{prediction_id}/{filename}')  # rename the file back after upload

            # Parse prediction labels and create a summary
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': float(l[1]),
                        'cy': float(l[2]),
                        'width': float(l[3]),
                        'height': float(l[4]),
                    } for l in labels]

                logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

                prediction_summary = {
                    'prediction_id': prediction_id,
                    'original_img_path': str(original_img_path),
                    'predicted_img_path': str(predicted_img_path),
                    'labels': labels,
                    'time': time.time()
                }

                json_data = json.dumps(prediction_summary)

                # TODO store the prediction_summary in a DynamoDB table
                dynamodb_table_name = 'Moshiko_Yolo'
                response = dynamodb_client.put_item(
                    TableName=dynamodb_table_name,
                    Item={
                        'prediction_id': {'S': prediction_id},  # Partition key
                        'ChatID': {'S': str(chat_id)},  # Sort key
                        'prediction_summary': {'S': json_data}
                    }
                )
                if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                    logger.info('Prediction summary stored in DynamoDB')
                else:
                    logger.error('Failed to store prediction summary in DynamoDB')

                # TODO perform a GET request to Polybot to `/results` endpoint

                service_name = 'poly-service:8443'
                endpoint = f'https://{service_name}/results/?predictionId={prediction_id}&chatId={chat_id}'

                try:
                    response = requests.get(endpoint, json=prediction_summary)
                    if response.status_code == 200:
                        logger.info('GET request to Polybot successful')
                    else:
                        logger.error('Failed to perform GET request to Polybot')
                except requests.RequestException as e:
                    logger.error(f'Error in GET request to Polybot: {e}')

            # Delete the message from the queue as the job is considered as DONE
            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)


if __name__ == "__main__":
    consume()
