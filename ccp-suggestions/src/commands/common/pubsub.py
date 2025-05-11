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

def pull_single_message(subscription_name):
    project_id = getenv('GOOGLE_CLOUD_PROJECT')
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_name)

    response = subscriber.pull(
        request={
            "subscription": subscription_path,
            "max_messages": 1,
        }
    )

    if not response.received_messages:
        return "No messages available."

    message = response.received_messages[0]
    data = message.message.data.decode('utf-8')
    attributes = message.message.attributes

    # Acknowledge the message so it's not redelivered
    subscriber.acknowledge(
        request={
            "subscription": subscription_path,
            "ack_ids": [message.ack_id],
        }
    )

    return {
        "message": message,
        "data": data,
        "attributes": attributes
    }