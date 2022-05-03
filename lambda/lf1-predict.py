import boto3
import json
import email
import logging
import datetime
from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
SAGEMAKER_ENDPOINT = '<SageMaker-Endpoint>'
VOCABULARY_LENGTH = 9013

def convert_date(date):
    dt_obj = datetime.datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %z')
    d = dt_obj.strftime('%B %d, %Y %I:%M %p')
    return d

def get_email(event):
    
    # extract email object key
    s3_record = event['Records'][0]['s3']
    bucket_name = s3_record['bucket']['name']
    object_key = s3_record['object']['key']
    
    # init s3 client and get email object
    client = boto3.client('s3')
    response = client.get_object(Bucket=bucket_name, Key=object_key)
    email_object = email.message_from_bytes(response['Body'].read())
    
    # extract email details
    sender = email_object.get('From')
    subject = email_object.get('Subject')
    date = convert_date(email_object.get('Date'))
    receiver = email_object.get('To')
    body = email_object.get_payload()[0].get_payload()
    format_body = body.strip('\n')
    format_body = [format_body.replace("=", "")]
    
    return sender, subject, date, receiver, format_body


def inference(body):
    
    encoded_message = one_hot_encode(body, VOCABULARY_LENGTH)
    vectorized_message = vectorize_sequences(encoded_message, VOCABULARY_LENGTH)
    data = json.dumps(vectorized_message.tolist())

    runtime = boto3.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(EndpointName=SAGEMAKER_ENDPOINT, ContentType='application/json', Body=data)
    json_response = json.loads(response["Body"].read())
    print("res:", json_response)
    
    if json_response['predicted_label'][0][0] == 0:
        label = 'NOT SPAM'
    else:
        label = 'SPAM'
    score = round(json_response['predicted_probability'][0][0], 4)
    score = score * 100
 
    return score, label


def send_response(score, label, sender, date, receiver, subject, body):

    email_body = f'We received your email sent at {date} with the subject {subject}.' + \
                 f'\n\nHere is a 240 character sample of the email body: \n\n{body[0:240]}' + \
                 f'\n\nThe email was categorized as {label} with {score}% confidence.'
    
    client = boto3.client('ses')
    
    if(sender):
        response = client.send_email(
            Source=receiver,
            Destination={
                'ToAddresses': [
                    sender,
                ]
            },
            Message={
                'Subject': {
                    'Data': 'Email Spam Predicter',
                },
                'Body': {
                    'Text': {
                        'Data': email_body,
                    },
                }
            },
        )


def lambda_handler(event, context):
   
    # get email in S3 bucket
    sender, subject, date, receiver, body = get_email(event)
    # call sagemaker
    score, label = inference(body)
    
    result = "\nscore:   " + str(score) + \
             "\nlabel:   " + str(label) + \
             "\nsender:  " + sender + \
             "\ndate:     " + date + \
             "\nsubject: " + subject + \
             "\nreceiver:" + receiver + \
             "\nbody:    " + body[0]
    
    send_response(str(score), str(label), sender, date, receiver, subject, body[0])
    
    #logger.debug("result={}".format(result))
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
