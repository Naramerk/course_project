import os
import sys
from pyspark.sql import SparkSession
import psycopg2
from psycopg2.extras import execute_values
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('parquet_to_postgres')

def get_postgres_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            dbname="reddit_analysis",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        raise

def create_tables(conn):
    """Create necessary tables if they don't exist"""
    try:
        with conn.cursor() as cur:
            # Create posts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.posts (
                    post_id VARCHAR(50) PRIMARY KEY,
                    author VARCHAR(100),
                    created_utc TIMESTAMP WITH TIME ZONE,
                    text TEXT,
                    subreddit VARCHAR(50),
                    upvotes INTEGER,
                    num_comments INTEGER,
                    parent_post_id VARCHAR(50),
                    sentiment_score FLOAT,
                    message_language VARCHAR(10),
                    country VARCHAR(10),
                    leader VARCHAR(100),
                    last_processed TIMESTAMP WITH TIME ZONE
                )
            """)
            conn.commit()
            logger.info("Tables created or verified")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        conn.rollback()
        raise

def transfer_data():
    """Transfer data from Parquet to PostgreSQL"""
    try:
        # Initialize Spark
        spark = SparkSession.builder \
            .appName("ParquetToPostgres") \
            .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.driver.host", "localhost") \
            .getOrCreate()
            
        # Read Parquet data
        parquet_path = "spark-warehouse/reddit_data"
        if not os.path.exists(parquet_path):
            logger.info("No new data to transfer")
            return
            
        df = spark.read.parquet(parquet_path)
        if df.count() == 0:
            logger.info("No records to transfer")
            return
            
        # Convert to list of tuples
        rows = df.collect()
        data = []
        for row in rows:
            data.append((
                row.post_id,
                row.author,
                row.created_utc,
                row.text,
                row.subreddit,
                row.upvotes,
                row.num_comments,
                row.parent_post_id,
                row.sentiment_score,
                row.message_language,
                row.country,
                row.leader,
                row.last_processed
            ))
            
        # Connect to PostgreSQL and transfer data
        conn = get_postgres_connection()
        create_tables(conn)
        
        with conn.cursor() as cur:
            # Insert data using execute_values for better performance
            execute_values(
                cur,
                """
                INSERT INTO public.posts (
                    post_id, author, created_utc, text, subreddit,
                    upvotes, num_comments, parent_post_id, sentiment_score,
                    message_language, country, leader, last_processed
                ) VALUES %s
                ON CONFLICT (post_id) DO UPDATE SET
                    upvotes = EXCLUDED.upvotes,
                    num_comments = EXCLUDED.num_comments,
                    last_processed = EXCLUDED.last_processed
                """,
                data
            )
            conn.commit()
            
        logger.info(f"Successfully transferred {len(data)} records to PostgreSQL")
        
    except Exception as e:
        logger.error(f"Error transferring data: {e}")
        if 'conn' in locals():
            conn.rollback()
        raise
        
    finally:
        if 'conn' in locals():
            conn.close()
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    try:
        transfer_data()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1) 