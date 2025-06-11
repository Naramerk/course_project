# Reddit Political Sentiment Analysis

This project collects and analyzes political discussions from Reddit, focusing on sentiment analysis and tracking mentions of countries and political leaders.

## Features

- Real-time collection of posts and comments from political subreddits
- Sentiment analysis of text content
- Detection of country and leader mentions
- Data storage in both Parquet format (via Spark) and PostgreSQL
- Automatic error recovery and retry mechanisms

## Prerequisites

- Python 3.8+
- Apache Spark
- PostgreSQL
- Reddit API credentials

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up PostgreSQL database:
   - Create a database named 'reddit_analysis'
   - Update connection settings in parquet_to_postgres.py if needed

## Configuration

1. Reddit API credentials are required. The default values are:
   - Client ID: eYMJ7Td4dqK3ROGLEWqgZQ
   - Client Secret: BOs69e4HSw_O3Q6Qa1YXAW_S8NwAIA

2. Update the credentials in get_reddit_token.py if needed.

## Usage

1. First, get a Reddit API token:
   ```bash
   python get_reddit_token.py
   ```

2. Run the collector:
   ```bash
   python reddit_collector_spark.py
   ```

The collector will:
- Fetch posts and comments from configured subreddits
- Process them for sentiment and entity detection
- Store data in Parquet format
- Transfer data to PostgreSQL

## Data Files

- countries.json: Contains country names and alternative references
- leaders.json: Contains political leader names and alternative references

## Error Handling

The system includes:
- Automatic retry mechanisms
- Spark session recovery
- Database connection error handling
- Token refresh handling 