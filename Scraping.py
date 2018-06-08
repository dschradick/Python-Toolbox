########## SCRAPING VON WEB-INHALTEN

### Mehrere Dateien
import glob
csv_files = glob.glob('sales*.csv')
frames = [pd.read_csv(f) for f in filesnames]
df = pd.concat(frames)


### File Downloading
from urllib.request import urlretrieve
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'


### GET request
import requests
url = "https://www.wikipedia.org"
r = requests.get(url)
text = r.text


### Webscraping mit BeautifulSoup
from bs4 import BeautifulSoup
html_doc = requests.get(url).text
soup = BeautifulSoup(html_doc,"lxml")
pretty_soup = soup.prettify()
print(soup.title,soup.get_text())
for link in soup.find_all('a'):
    print(link.get('href'))


### JSON & APIs
import json, requests  # tweepy für
with open('data.json','r') as json_file
    json_data = json.load(json_file) # => liefert dict
url = 'http://www.omdbapi.com?apikey=ff21610b&t=hackers'
json_data = requests.get(url).json()
for k in json_data.keys():
    print(k + ': ', json_data[k])


### Twitter
import tweepy, json
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.tweet_list = []
        self.file = open("tweets.txt", "w")
    def on_status(self, status):
        tweet = status._json
        self.file.write(json.dumps(tweet) + '\n')
        print("Tweet: " + tweet['text'])
        self.tweet_list.append(status)
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
            self.file.close()

access_token = ""; access_token_secret = ""; consumer_key = ""; consumer_secret = ""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
l = MyStreamListener()
stream = tweepy.Stream(auth, l)
stream.filter(track=['data science', 'mmo','clinton','trump'])


# Verabeitung mit Pandas
import pandas as pd
import json
tweets_data = []
with open('tweets.txt','r') as json_file:
    for line in json_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue

tweets = pd.DataFrame()
tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
