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
import pandas as pd

# Add VADER sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reddit_collector_spark_process')

# Set Spark home
os.environ['SPARK_HOME'] = '/opt/homebrew/opt/apache-spark/libexec'

class SparkProcessRedditCollector:
    """
    Reddit collector that use of Spark for all processing - no local fallbacks
    """
    
    def __init__(self):
        """Initialize the Reddit collector with Spark processing"""
        logger.info("Initializing Spark Reddit collector")
        
        # Initialize Spark session with optimized settings
        self.spark = SparkSession.builder \
            .appName("RedditCollectorProcessMode") \
            .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.host", "localhost") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.default.parallelism", "4") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
            
        # Set log level to reduce noise
        self.spark.sparkContext.setLogLevel("WARN")
            
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
        
        # Load reference data and setup UDFs
        self.load_reference_data()
        self.setup_spark_udfs()
        
        # Load token
        logger.info("Loading Reddit token")
        self.token_info = self.load_token()
        if not self.token_info:
            raise ValueError("No valid token found. Please run get_reddit_token.py first.")
        logger.info("Successfully loaded Reddit token")
        
        # Subreddits to monitor
        self.subreddits = [
            'worldnews', 'politics', 'europe', 'geopolitics',
            'news', 'GlobalNews', 'InternationalNews', 'Diplomacy',
            'de', 'france', 'ukpolitics'
        ]
        
        # Pagination tracking - track last processed post ID per subreddit
        self.last_post_ids = {}
        # Track last fetch time to avoid re-processing old posts
        self.last_fetch_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Start from 1 hour ago
        
        # Create tables
        self.create_tables()
        
        logger.info("SPARK Reddit collector initialization complete")
        
    def load_reference_data(self):
        """Load countries and leaders data for UDF processing"""
        try:
            # Load countries
            countries_file = os.path.join(os.path.dirname(__file__), 'data', 'countries.json')
            with open(countries_file, 'r') as f:
                countries_data = json.load(f)
            
            # Load leaders
            leaders_file = os.path.join(os.path.dirname(__file__), 'data', 'leaders.json')
            with open(leaders_file, 'r') as f:
                leaders_data = json.load(f)
                
            # Create lookup dictionaries that will be recreated in each UDF
            self.country_lookup = {}
            for country in countries_data:
                name = country['name'].lower()
                code = country['code']
                self.country_lookup[name] = code
                for alt in country.get('alternatives', []):
                    self.country_lookup[alt.lower()] = code
                    
            self.leader_lookup = {}
            for leader in leaders_data:
                name = leader['name'].lower()
                self.leader_lookup[name] = leader['name']
                for alt in leader.get('alternatives', []):
                    self.leader_lookup[alt.lower()] = leader['name']
                    
            # Bot patterns
            self.bot_patterns = [
                r'bot$', r'^bot', r'_bot_', r'^auto', 
                r'moderator', r'transcriber', r'translator'
            ]
            
            self.known_bots = [
                'AutoModerator', 'BotDefense', 'RepostSleuthBot', 'transcribot',
                'translator-BOT', 'reddit-stream', 'RemindMeBot', 'converter-bot',
                'TweetPoster', 'WikiTextBot', 'CommonMisspellingBot', 'LinkFixerBot',
                'TotesMessenger', 'GoodBot_BadBot', 'MAGIC_EYE_BOT', 'HelperBot_'
            ]
            
            logger.info(f"Loaded {len(countries_data)} countries and {len(leaders_data)} leaders for Spark UDFs")
            
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            raise
            
    def setup_spark_udfs(self):
        """Setup Spark UDFs for processing"""
        
        # Get data for UDFs (will be recreated in each UDF call)
        country_lookup = self.country_lookup
        leader_lookup = self.leader_lookup
        bot_patterns = self.bot_patterns
        known_bots = self.known_bots
        
        # Sentiment analysis UDF
        def analyze_sentiment_udf_func(text):
            if not text or len(text.strip()) == 0:
                return 0.0
            try:
                # Initialize VADER on worker (each call)
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                
                # Clean text
                text_cleaned = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text_cleaned = re.sub(r'@\w+|#\w+', '', text_cleaned)
                text_cleaned = text_cleaned.strip()
                
                if len(text_cleaned) == 0:
                    return 0.0
                    
                scores = analyzer.polarity_scores(text_cleaned)
                return float(scores['compound'])
            except Exception:
                return 0.0
                
        # Language detection UDF
        def detect_language_udf_func(text):
            if not text or len(text.strip()) == 0:
                return 'en'
            try:
                from langdetect import detect
                return detect(text.strip())
            except:
                return 'en'
                
        # Country detection UDF
        def detect_country_udf_func(text):
            if not text:
                return None
            try:
                # Recreate lookup in UDF
                country_lookup = {
                    'usa': 'US', 'america': 'US', 'united states': 'US',
                    'russia': 'RU', 'russian': 'RU', 'putin': 'RU',
                    'china': 'CN', 'chinese': 'CN', 'xi jinping': 'CN',
                    'germany': 'DE', 'german': 'DE', 'merkel': 'DE',
                    'france': 'FR', 'french': 'FR', 'macron': 'FR',
                    'ukraine': 'UA', 'ukrainian': 'UA', 'zelensky': 'UA',
                    'uk': 'GB', 'britain': 'GB', 'british': 'GB'
                }
                
                text_lower = text.lower()
                for keyword, code in country_lookup.items():
                    if keyword in text_lower:
                        return code
                return None
            except Exception:
                return None
                
        # Leader detection UDF
        def detect_leader_udf_func(text):
            if not text:
                return None
            try:
                # Recreate lookup in UDF
                leader_lookup = {
                    'biden': 'Joe Biden', 'trump': 'Donald Trump',
                    'putin': 'Vladimir Putin', 'xi jinping': 'Xi Jinping',
                    'macron': 'Emmanuel Macron', 'zelensky': 'Volodymyr Zelensky',
                    'scholz': 'Olaf Scholz', 'sunak': 'Rishi Sunak'
                }
                
                text_lower = text.lower()
                for keyword, name in leader_lookup.items():
                    if keyword in text_lower:
                        return name
                return None
            except Exception:
                return None
                
        # Bot detection UDF
        def is_bot_udf_func(author):
            if not author:
                return True
            try:
                author_lower = author.lower()
                
                # Known bots check
                known_bots = [
                    'automoderator', 'botdefense', 'repostsleuthbot', 'transcribot',
                    'translator-bot', 'reddit-stream', 'remindmebot', 'converter-bot'
                ]
                
                if author_lower in known_bots:
                    return True
                    
                # Pattern check
                bot_patterns = [r'bot$', r'^bot', r'_bot_', r'^auto', r'moderator']
                for pattern in bot_patterns:
                    if re.search(pattern, author, re.IGNORECASE):
                        return True
                        
                return False
            except Exception:
                return False
                
        # Register UDFs
        self.sentiment_udf = udf(analyze_sentiment_udf_func, FloatType())
        self.language_udf = udf(detect_language_udf_func, StringType())
        self.country_udf = udf(detect_country_udf_func, StringType())
        self.leader_udf = udf(detect_leader_udf_func, StringType())
        self.is_bot_udf = udf(is_bot_udf_func, BooleanType())
        
        # Batch UDF for larger datasets
        from pyspark.sql.functions import pandas_udf
        
        @pandas_udf(returnType=StructType([
            StructField("sentiment_scores", ArrayType(FloatType()), True),
            StructField("languages", ArrayType(StringType()), True)
        ]))
        def batch_analysis_udf(texts: pd.Series) -> pd.DataFrame:
            """Batch process sentiment and language in one UDF"""
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                from langdetect import detect
                
                analyzer = SentimentIntensityAnalyzer()
                sentiments = []
                languages = []
                
                for text in texts:
                    if not text or len(str(text).strip()) == 0:
                        sentiments.append(0.0)
                        languages.append('en')
                        continue
                        
                    try:
                        text_str = str(text)
                        # Clean text
                        text_cleaned = re.sub(r'http\S+|www\S+|https\S+', '', text_str, flags=re.MULTILINE)
                        text_cleaned = re.sub(r'@\w+|#\w+', '', text_cleaned)
                        text_cleaned = text_cleaned.strip()
                        
                        if len(text_cleaned) == 0:
                            sentiments.append(0.0)
                            languages.append('en')
                        else:
                            # Sentiment
                            scores = analyzer.polarity_scores(text_cleaned)
                            sentiments.append(float(scores['compound']))
                            
                            # Language
                            try:
                                lang = detect(text_cleaned)
                            except:
                                lang = 'en'
                            languages.append(lang)
                    except Exception:
                        sentiments.append(0.0)
                        languages.append('en')
                        
                return pd.DataFrame({
                    'sentiment_scores': [sentiments],
                    'languages': [languages]
                })
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                return pd.DataFrame({
                    'sentiment_scores': [[0.0] * len(texts)],
                    'languages': [['en'] * len(texts)]
                })
                
        self.batch_analysis_udf = batch_analysis_udf
        
        logger.info("Successfully set up Spark UDFs")
        
    def create_tables(self):
        """Create Spark tables if they don't exist"""
        try:
            empty_df = self.spark.createDataFrame([], self.reddit_schema)
            empty_df.write \
                .mode("ignore") \
                .parquet("spark-warehouse/reddit_data")
            logger.info("Created or verified reddit_data table")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
            
    def load_token(self):
        """Load the Reddit token from file"""
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
        """Get headers for API requests"""
        return {
            'User-Agent': 'Political Sentiment Analysis/1.0',
            'Authorization': f"Bearer {self.token_info['access_token']}"
        }
        
    def fetch_posts(self, subreddit, after=None, limit=100):
        """Fetch posts from a subreddit using Reddit API"""
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
                    'parent_post_id': None,
                    'sentiment_score': None,  # Will be filled by UDF
                    'message_language': None,  # Will be filled by UDF
                    'country': None,  # Will be filled by UDF
                    'leader': None,  # Will be filled by UDF
                    'last_processed': None  # Will be filled by UDF
                }
                
                posts.append(post_dict)
                
            logger.info(f"Retrieved {len(posts)} posts from r/{subreddit}")
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error retrieving posts from r/{subreddit}: {e}")
            
        return posts
        
    def fetch_comments(self, post_id, limit=100):
        """Fetch comments for a post using Reddit API"""
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
                        'parent_post_id': post_id,
                        'sentiment_score': None,  # Will be filled by UDF
                        'message_language': None,  # Will be filled by UDF
                        'country': None,  # Will be filled by UDF
                        'leader': None,  # Will be filled by UDF
                        'last_processed': None  # Will be filled by UDF
                    }
                    
                    comments.append(comment_dict)
                    
            logger.info(f"Retrieved {len(comments)} comments for post {post_id}")
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error retrieving comments for post {post_id}: {e}")
            
        return comments
        
    def process_with_spark_process(self, items, process_batch=False):
        if not items:
            return None
            
        try:
            logger.info(f"🚀 Spark processing for {len(items)} items")
            
            # Create DataFrame from raw data
            df = self.spark.createDataFrame(items, self.reddit_schema)
            
            # ALWAYS filter bots using Spark UDF
            df_filtered = df.filter(~self.is_bot_udf(col("author")))
            
            record_count = df_filtered.count()
            logger.info(f"After bot filtering: {record_count} records")
            
            if record_count == 0:
                return None
            
            if process_batch or record_count > 15:
                # Use batch processing - but simplified to avoid arrays_zip issues
                logger.info("⚡ Using SIMPLIFIED BATCH processing (individual UDFs)")
                
                # Temporarily disable complex batch processing due to arrays_zip issues
                # Use individual UDFs instead for stability
                df_final = df_filtered \
                    .withColumn("sentiment_score", self.sentiment_udf(col("text"))) \
                    .withColumn("message_language", self.language_udf(col("text")))
                
            else:
                # Use individual UDFs
                logger.info("⚡ Using INDIVIDUAL UDF processing")
                df_final = df_filtered \
                    .withColumn("sentiment_score", self.sentiment_udf(col("text"))) \
                    .withColumn("message_language", self.language_udf(col("text")))
            
            # Apply entity detection (always individual for accuracy)
            df_final = df_final \
                .withColumn("country", self.country_udf(col("text"))) \
                .withColumn("leader", self.leader_udf(col("text"))) \
                .withColumn("last_processed", lit(datetime.now(timezone.utc)))
            
            # Process materialization and cache
            df_final = df_final.cache()
            final_count = df_final.count()
            
            logger.info(f"✅ SPARK processed {final_count} items successfully")
            
            return df_final
            
        except Exception as e:
            logger.error(f"❌ Error in Spark processing: {e}")
            raise  # No fallback - fail if Spark fails
            
    def save_to_spark(self, df):
        """Save DataFrame to Spark storage"""
        try:
            df.write \
                .mode("append") \
                .parquet("spark-warehouse/reddit_data")
                
            logger.info(f"Saved {df.count()} records to Spark storage")
            
        except Exception as e:
            logger.error(f"Error saving to Spark: {e}")
            raise
            
    def process_batch_and_transfer(self, process_batch=False):
        """Process a batch of Reddit data and transfer to PostgreSQL"""
        try:
            batch_count = 0
            all_items = []
            new_posts_found = False
            
            # Collect from all subreddits
            for subreddit in self.subreddits:
                # Use pagination to get newer posts
                after = self.last_post_ids.get(subreddit)
                logger.info(f"Fetching from r/{subreddit}, after post: {after}")
                
                subreddit_posts = self.fetch_posts(subreddit, limit=1)
                
                if subreddit_posts:
                    # Filter posts that are newer than our last fetch time
                    filtered_posts = []
                    for post in subreddit_posts:
                        if post['created_utc'] > self.last_fetch_time:
                            filtered_posts.append(post)
                            new_posts_found = True
                    
                    if filtered_posts:
                        # Update the last post ID for pagination (use the newest post)
                        self.last_post_ids[subreddit] = filtered_posts[0]['post_id']
                        all_items.extend(filtered_posts)
                        
                        # Get comments for new posts only
                        for post in filtered_posts:
                            post_comments = self.fetch_comments(post['post_id'], limit=3)  
                            all_items.extend(post_comments)
                        
                        logger.info(f"Found {len(filtered_posts)} new posts from r/{subreddit}")
                    else:
                        logger.info(f"No new posts from r/{subreddit}")
                else:
                    logger.info(f"No posts fetched from r/{subreddit}")
                    
                batch_count += 1
                if batch_count >= 3:  # Process more subreddits 
                    break
            
            # Update last fetch time
            self.last_fetch_time = datetime.now(timezone.utc)
            
            # Spark processing - no local alternative
            if all_items:
                # Calculate counts separately to avoid Spark column errors
                post_count = 0
                comment_count = 0
                for item in all_items:
                    if item.get('parent_post_id') is None:
                        post_count += 1
                    else:
                        comment_count += 1
                logger.info(f"🚀 FORCING Spark processing for {len(all_items)} items ({post_count} posts, {comment_count} comments)")
                
                processed_df = self.process_with_spark_process(all_items, process_batch=process_batch)
                if processed_df is not None:
                    self.save_to_spark(processed_df)
                    
                    # Show sample results
                    logger.info("📊 Sample of SPARK processed data:")
                    processed_df.select("post_id", "author", "sentiment_score", "country", "leader") \
                              .show(5, truncate=False)
                else:
                    logger.warning("No data survived Spark processing (all bots filtered)")
                
                logger.info(f"✅ SPARK processed {len(all_items)} total items")
            else:
                if new_posts_found:
                    logger.warning("New posts found but no items to process after filtering")
                else:
                    logger.info("No new posts found - skipping processing")
                return False
            
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
            raise  # No fallback - fail hard
            
    def collect_continuously(self, interval=5, process_batch=False):
        """Continuously collect and process data using ONLY Spark"""
        logger.info("🚀 Starting SPARK continuous collection")
        
        while True:
            try:
                if self.token_info is None:
                    logger.error("No valid token available. Please refresh using get_reddit_token.py")
                    time.sleep(interval)
                    continue
                    
                # Process one batch - Spark only
                success = self.process_batch_and_transfer(process_batch=process_batch)
                if not success:
                    logger.error("Failed to process batch with Spark")
                    
                logger.info(f"⏱️  Waiting {interval} seconds before next SPARK batch...")
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in collection cycle: {e}")
                logger.info("Restarting Spark session...")
                
                try:
                    self.spark.stop()
                except:
                    pass
                    
                try:
                    # Reinitialize
                    self.__init__()
                except Exception as spark_error:
                    logger.error(f"Failed to restart Spark: {spark_error}")
                    break  # Exit if can't restart Spark
                    
                time.sleep(interval)
                
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'spark'):
                self.spark.stop()
                logger.info("🧹 Spark session stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Spark Reddit collector")
    parser.add_argument("--interval", type=int, default=5, help="Interval between runs in seconds (default: 5)")
    parser.add_argument("--process-batch", action="store_true", help="Process use of batch UDF processing always")
    args = parser.parse_args()
    
    collector = None
    try:
        logger.info("🚀 Starting SPARK Reddit collector - NO LOCAL PROCESSING")
        collector = SparkProcessRedditCollector()
        logger.info(f"Starting continuous collection with {args.interval}s interval")
        
        collector.collect_continuously(interval=args.interval, process_batch=args.process_batch)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Spark collector stopped by user")
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}", exc_info=True)
    finally:
        if collector:
            collector.cleanup()
        logger.info("🏁 Spark collector stopped") 
