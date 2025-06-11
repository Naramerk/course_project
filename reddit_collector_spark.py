import os
import sys
import requests
import time
from datetime import datetime, timezone, timedelta
import json
import re
from langdetect import detect
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import logging
import subprocess
import shutil

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reddit_collector_spark')

# Set Spark home
os.environ['SPARK_HOME'] = '/opt/homebrew/opt/apache-spark/libexec'

class SentimentAnalyzer:
    """Simple sentiment analyzer returning values between -1 and 1"""
    def analyze_sentiment(self, text):
        """Returns a tuple of (sentiment_score, language)
        sentiment_score is between -1 and 1
        """
        try:
            # Simple sentiment analysis based on positive/negative word counts
            positive_words = {'good', 'great', 'awesome', 'excellent', 'happy', 'positive', 'wonderful', 'love', 'best'}
            negative_words = {'bad', 'terrible', 'awful', 'horrible', 'sad', 'negative', 'worst', 'hate', 'poor'}
            
            words = text.lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            if pos_count == 0 and neg_count == 0:
                score = 0.0
            else:
                total = pos_count + neg_count
                score = (pos_count - neg_count) / (pos_count + neg_count)
            
            # Detect language
            try:
                lang = detect(text)
            except:
                lang = 'en'
                
            return score, lang
        except:
            return 0.0, 'en'

class RedditSparkCollector:
    """
    Collector for Reddit posts and comments using Reddit's API with Apache Spark processing
    """
    
    def __init__(self):
        """
        Initialize the Reddit collector with Spark
        """
        logger.info("Initializing Reddit Spark collector")
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("RedditCollector") \
            .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.host", "localhost") \
            .getOrCreate()
            
        # Define schema for Reddit data
        self.reddit_schema = StructType([
            StructField("post_id", StringType(), False),
            StructField("author", StringType(), True),
            StructField("created_utc", TimestampType(), False),
            StructField("text", StringType(), True),
            StructField("subreddit", StringType(), True),
            StructField("upvotes", IntegerType(), True),
            StructField("num_comments", IntegerType(), True),
            StructField("parent_post_id", StringType(), True),
            StructField("sentiment_score", FloatType(), True),
            StructField("message_language", StringType(), True),
            StructField("country", StringType(), True),
            StructField("leader", StringType(), True),
            StructField("last_processed", TimestampType(), True)
        ])
        
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("Initialized sentiment analyzer")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}", exc_info=True)
            raise
            
        self.process_name = 'reddit_collector_spark'
        
        # Load token
        logger.info("Loading Reddit token")
        self.token_info = self.load_token()
        if not self.token_info:
            raise ValueError("No valid token found. Please run get_reddit_token.py first.")
        logger.info("Successfully loaded Reddit token")
            
        # Common bot usernames and patterns
        self.bot_patterns = [
            r'bot$', r'^bot', r'_bot_', r'^auto', 
            r'moderator', r'transcriber', r'translator'
        ]
        
        # List of known bot usernames
        self.known_bots = [
            'AutoModerator', 'BotDefense', 'RepostSleuthBot', 'transcribot',
            'translator-BOT', 'reddit-stream', 'RemindMeBot', 'converter-bot',
            'TweetPoster', 'WikiTextBot', 'CommonMisspellingBot', 'LinkFixerBot',
            'TotesMessenger', 'GoodBot_BadBot', 'MAGIC_EYE_BOT', 'HelperBot_'
        ]
        
        # Subreddits to monitor
        self.subreddits = [
            'worldnews', 'politics', 'europe', 'geopolitics',
            'news', 'GlobalNews', 'InternationalNews', 'Diplomacy',
            'de', 'france', 'ukpolitics'
        ]
        
        # Load countries and leaders
        logger.info("Loading countries and leaders data")
        self.load_countries()
        self.load_leaders()
        logger.info("Successfully loaded countries and leaders data")
        
        # Create or get tables
        self.create_tables()
        
        logger.info("Reddit Spark collector initialization complete")
        
    def create_tables(self):
        """
        Create Spark tables if they don't exist
        """
        try:
            # Create empty DataFrame with schema
            empty_df = self.spark.createDataFrame([], self.reddit_schema)
            
            # Create table if not exists
            empty_df.write \
                .mode("ignore") \
                .parquet("spark-warehouse/reddit_data")
                
            logger.info("Created or verified reddit_data table")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
            
    def load_token(self):
        """
        Load the Reddit token from file
        """
        try:
            token_file = os.path.join(os.path.dirname(__file__), 'reddit_token.json')
            with open(token_file) as f:
                token_info = json.load(f)
                
            # Check if token is expired
            retrieved_at = datetime.fromisoformat(token_info['retrieved_at'])
            expires_in = token_info['expires_in']
            if datetime.now(timezone.utc) > retrieved_at + timedelta(seconds=expires_in):
                logger.warning("Token is expired. Please refresh it using get_reddit_token.py --refresh")
                return None
                
            return token_info
            
        except Exception as e:
            logger.error(f"Error loading token: {e}")
            return None
            
    def get_headers(self):
        """
        Get headers for API requests
        """
        return {
            'User-Agent': 'Political Sentiment Analysis/1.0',
            'Authorization': f"Bearer {self.token_info['access_token']}"
        }
        
    def load_countries(self):
        """
        Load countries from JSON file
        """
        try:
            countries_file = os.path.join(os.path.dirname(__file__), 'data', 'countries.json')
            with open(countries_file, 'r') as f:
                self.countries = json.load(f)
            logger.info(f"Loaded {len(self.countries)} countries")
        except Exception as e:
            logger.error(f"Error loading countries: {e}")
            self.countries = []
            
    def load_leaders(self):
        """
        Load leaders from JSON file
        """
        try:
            leaders_file = os.path.join(os.path.dirname(__file__), 'data', 'leaders.json')
            with open(leaders_file, 'r') as f:
                self.leaders = json.load(f)
            logger.info(f"Loaded {len(self.leaders)} leaders")
        except Exception as e:
            logger.error(f"Error loading leaders: {e}")
            self.leaders = []
            
    def fetch_posts(self, subreddit, after=None, limit=100):
        """
        Fetch posts from a subreddit using Reddit API
        """
        posts = []
        
        try:
            url = f"https://oauth.reddit.com/r/{subreddit}/new"
            params = {
                'limit': limit,
                'raw_json': 1
            }
            
            if after:
                params['after'] = f"t3_{after}"
                
            response = requests.get(
                url,
                params=params,
                headers=self.get_headers()
            )
            response.raise_for_status()
            
            data = response.json()['data']['children']
            
            for post in data:
                post_data = post['data']
                post_dict = {
                    'post_id': post_data['id'],
                    'author': post_data.get('author', '[deleted]'),
                    'created_utc': datetime.fromtimestamp(post_data['created_utc'], tz=timezone.utc),
                    'text': post_data.get('selftext', '') if post_data.get('selftext') else post_data.get('title', ''),
                    'subreddit': subreddit,
                    'upvotes': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'parent_post_id': None
                }
                
                posts.append(post_dict)
                
            logger.info(f"Retrieved {len(posts)} posts from r/{subreddit}")
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error retrieving posts from r/{subreddit}: {e}")
            
        return posts
        
    def fetch_comments(self, post_id, limit=100):
        """
        Fetch comments for a post using Reddit API
        """
        comments = []
        
        try:
            url = f"https://oauth.reddit.com/comments/{post_id}"
            params = {
                'limit': limit,
                'depth': 1,
                'raw_json': 1
            }
            
            response = requests.get(
                url,
                params=params,
                headers=self.get_headers()
            )
            response.raise_for_status()
            
            if len(response.json()) > 1:
                data = response.json()[1]['data']['children']
                
                for comment in data:
                    if comment['kind'] != 't1':
                        continue
                        
                    comment_data = comment['data']
                    comment_dict = {
                        'post_id': comment_data['id'],
                        'author': comment_data.get('author', '[deleted]'),
                        'created_utc': datetime.fromtimestamp(comment_data['created_utc'], tz=timezone.utc),
                        'text': comment_data.get('body', ''),
                        'subreddit': comment_data.get('subreddit', ''),
                        'upvotes': comment_data.get('score', 0),
                        'num_comments': 0,
                        'parent_post_id': post_id
                    }
                    
                    comments.append(comment_dict)
                    
            logger.info(f"Retrieved {len(comments)} comments for post {post_id}")
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error retrieving comments for post {post_id}: {e}")
            
        return comments
        
    def is_bot(self, author):
        """
        Check if a user is likely a bot based on username patterns
        """
        if author.lower() in [bot.lower() for bot in self.known_bots]:
            return True
            
        for pattern in self.bot_patterns:
            if re.search(pattern, author, re.IGNORECASE):
                return True
                
        return False
        
    def detect_country_and_leader(self, text):
        """
        Detect mentions of countries and leaders in text
        """
        text_lower = text.lower()
        
        country_code = None
        for country in self.countries:
            name = country['name'].lower()
            alternatives = [alt.lower() for alt in country.get('alternatives', [])]
            if name in text_lower or any(alt in text_lower for alt in alternatives):
                country_code = country['code']
                break
                
        leader_name = None
        for leader in self.leaders:
            name = leader['name'].lower()
            alternatives = [alt.lower() for alt in leader.get('alternatives', [])]
            if name in text_lower or any(alt in text_lower for alt in alternatives):
                leader_name = leader['name']
                break
                
        return country_code, leader_name
        
    def process_items_spark(self, items):
        """
        Process items using Spark
        """
        if not items:
            return None
            
        # Process items locally first
        processed_items = []
        for item in items:
            # Check if author is a bot
            if self.is_bot(item['author']):
                continue
                
            # Process sentiment and language
            sentiment_score, language = self.sentiment_analyzer.analyze_sentiment(item['text'])
            
            # Detect country and leader
            country, leader = self.detect_country_and_leader(item['text'])
            
            # Add processed fields
            item['sentiment_score'] = sentiment_score
            item['message_language'] = language
            item['country'] = country
            item['leader'] = leader
            item['last_processed'] = datetime.now(timezone.utc)
            
            processed_items.append(item)
            
        if not processed_items:
            return None
            
        # Convert processed items to Spark DataFrame
        return self.spark.createDataFrame(processed_items, self.reddit_schema)
        
    def save_to_spark(self, df):
        """
        Save DataFrame to Spark storage
        """
        try:
            df.write \
                .mode("append") \
                .parquet("spark-warehouse/reddit_data")
                
            logger.info(f"Saved {df.count()} records to Spark storage")
            
        except Exception as e:
            logger.error(f"Error saving to Spark: {e}")
            raise
            
    def get_last_processed_timestamp(self):
        """
        Get the timestamp of the last processed item
        """
        try:
            df = self.spark.read.parquet("spark-warehouse/reddit_data")
            if df.count() > 0:
                last_timestamp = df.agg(max("created_utc").alias("last_timestamp")).first().last_timestamp
                return last_timestamp
            return None
            
        except Exception as e:
            logger.error(f"Error getting last timestamp: {e}")
            return None
            
    def process_batch_and_transfer(self):
        """Process a batch of Reddit data and transfer to PostgreSQL"""
        try:
            batch_count = 0
            posts = []
            comments = []
            
            # Collect from all subreddits
            for subreddit in self.subreddits:
                subreddit_posts = self.fetch_posts(subreddit, limit=60)  # Reduced from 10 to 3
                posts.extend(subreddit_posts)
                
                for post in subreddit_posts:
                    post_comments = self.fetch_comments(post['post_id'], limit=2)  # Reduced from 10 to 2
                    comments.extend(post_comments)
                    
                batch_count += 1
                if batch_count >= 60:  # Reduced from 10 to 3 subreddits per batch
                    break
                    
            # Process and save posts
            if posts:
                posts_df = self.process_items_spark(posts)
                if posts_df is not None:
                    self.save_to_spark(posts_df)
                
            # Process and save comments
            if comments:
                comments_df = self.process_items_spark(comments)
                if comments_df is not None:
                    self.save_to_spark(comments_df)
                
            logger.info(f"Processed {len(posts)} posts and {len(comments)} comments")
            
            # Transfer to PostgreSQL
            logger.info("Transferring data to PostgreSQL")
            subprocess.run(['python', 'parquet_to_postgres.py'], check=True)
            
            # Clean up Spark warehouse
            logger.info("Cleaning up Spark warehouse")
            warehouse_path = "spark-warehouse/reddit_data"
            if os.path.exists(warehouse_path):
                shutil.rmtree(warehouse_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return False
            
    def collect_continuously(self, interval=3):
        """
        Continuously collect and process data from monitored subreddits
        """
        while True:
            try:
                if self.token_info is None:
                    logger.error("No valid token available. Please refresh using get_reddit_token.py")
                    time.sleep(interval)
                    continue
                    
                # Process one batch
                success = self.process_batch_and_transfer()
                if not success:
                    logger.error("Failed to process batch, waiting before retry")
                    
                logger.info(f"Waiting {interval} seconds before next batch...")
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in collection cycle: {e}")
                logger.info("Restarting Spark session...")
                try:
                    self.spark.stop()
                except:
                    pass
                    
                try:
                    # Reinitialize Spark session
                    self.spark = SparkSession.builder \
                        .appName("RedditCollector") \
                        .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
                        .config("spark.executor.memory", "2g") \
                        .config("spark.driver.memory", "2g") \
                        .config("spark.driver.host", "localhost") \
                        .getOrCreate()
                except Exception as spark_error:
                    logger.error(f"Failed to restart Spark: {spark_error}")
                    
                time.sleep(interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Reddit Spark collector script")
    parser.add_argument("--interval", type=int, default=3, help="Interval between runs in seconds (default: 3)")
    args = parser.parse_args()
    
    while True:
        try:
            logger.info("Starting Reddit Spark collector")
            collector = RedditSparkCollector()
            logger.info(f"Starting continuous collection with {args.interval}s interval")
            
            collector.collect_continuously(interval=args.interval)
        except KeyboardInterrupt:
            logger.info("Script stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            logger.info("Restarting collector in 5 seconds...")
            time.sleep(5)
            continue 