from json import dumps
from os import getenv
from google.cloud import pubsub_v1

def publish_message(topic_name, message: dict):
    project_id = getenv('GOOGLE_CLOUD_PROJECT')
    topic = topic_name

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic)

    message_bytes = dumps(message).encode('utf-8')

    attributes = {
        "content-type": "application/json"
    }
    future = publisher.publish(topic_path, data=message_bytes, **attributes)
    return f"Published message ID: {future.result()}"