from src.constants.urls import rabbit_mq_url
import pika

def callback(ch,method,properties,body):
    print('\n [x] Recieved a new video. Processing')

if __name__ == "__main__":
    print('Initializing video processing worker')
    print(f'Trying to connect with {rabbit_mq_url}')
    credentials = pika.PlainCredentials(username='guest', password='guest')
    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbit_mq_url, '/', credentials))
    channel = connection.channel()
    channel.queue_declare(queue='processing', durable= True)
    channel.basic_consume(queue='processing', auto_ack=False, on_message_callback=callback)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
    print('Queue closed')