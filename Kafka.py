########## KAFKA - BASICS
# Verzeichnis: /usr/hdp/current/kafka-broker/bin

## Neues Topic erzeugen
# ./kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic topic1

## Topic anzeigen
# ./kafka-topics.sh --list --zookeeper localhost:2181

## Konsole: Producer
# => kann auch als gateway benutzt werden
# ./kafka-console-producer.sh --broker-list sandbox-hdp.hortonworks.com:6667 --topic topic1

## Konsole: Consumer
# ./kafka-console-consumer.sh --bootstrap-server sandbox-hdp.hortonworks.com:6667 --topic topic1 --from-beginning

# spark-streaming-receiver
# spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.0 pyspark_receiver.py sandbox-hdp.hortonworks.com:6667 topic1


### Einfacher Producer
from confluent_kafka import Producer
p = Producer({'bootstrap.servers': 'sandbox-hdp.hortonworks.com:6667'})
p.produce('topic1', "Dies ist die Nachricht")
p.flush(30)


### Konsumer
from confluent_kafka import Consumer, KafkaError

settings = {
    'bootstrap.servers': 'sandbox-hdp.hortonworks.com:6667',
    'group.id': 'mygroup', 'client.id': 'client1',
    'enable.auto.commit': True, 'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'}
}

c = Consumer(settings)
c.subscribe(['mytopic'])

try:
    while True:
        msg = c.poll(0.1)
        if msg is None:
            continue
        elif not msg.error():
            print('Received message: {0}'.format(msg.value()))
        elif msg.error().code() == KafkaError._PARTITION_EOF:
            print('End of partition reached {0}/{1}'
                  .format(msg.topic(), msg.partition()))
        else:
            print('Error occured: {0}'.format(msg.error().str()))
except KeyboardInterrupt:
    pass
finally:
    c.close()
