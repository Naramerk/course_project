import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy.stats import gaussian_kde
import pycountry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database connection
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'reddit_analysis')
DB_USER = os.getenv('POSTGRES_USER', 'nargiza')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'your_password')

# Create database connection
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

def normalize_country_name(name):
    """Normalize country names to avoid duplicates"""
    if not name:
        return name
        
    name = name.strip().upper()
    
    country_mapping = {
        'US': 'USA',
        'USA': 'USA',
        'UNITED STATES': 'USA',
        'UNITED STATES OF AMERICA': 'USA',
        'UK': 'United Kingdom',
        'GB': 'United Kingdom',
        'GREAT BRITAIN': 'United Kingdom',
        'UNITED KINGDOM': 'United Kingdom',
        'RUSSIA': 'Russia',
        'RF': 'Russia',
        'RUS': 'Russia',
        'RUSSIAN FEDERATION': 'Russia',
        'CHINA': 'China',
        'PRC': 'China',
        "PEOPLE'S REPUBLIC OF CHINA": 'China',
        'UAE': 'United Arab Emirates',
        'UNITED ARAB EMIRATES': 'United Arab Emirates',
        'EU': 'European Union',
        'EUROPEAN UNION': 'European Union',
    }
    
    return country_mapping.get(name, name.title())

def normalize_leader_name(name):
    """Normalize leader names to avoid duplicates"""
    name_mapping = {
        'Macron': 'Emmanuel Macron',
        'Sunak': 'Rishi Sunak',
        'Trump': 'Donald Trump',
        'Biden': 'Joe Biden',
        'Putin': 'Vladimir Putin',
        'Xi': 'Xi Jinping'
    }
    return name_mapping.get(name, name)

def load_data():
    """Load and preprocess data from database"""
    try:
        query = """
        SELECT post_id, text, sentiment_score, country, leader, upvotes, created_utc,
               EXTRACT(HOUR FROM created_utc) as hour_of_day,
               EXTRACT(DOW FROM created_utc) as day_of_week
        FROM posts 
        WHERE sentiment_score IS NOT NULL
        """
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} records from database")
        
        if len(df) == 0:
            logger.warning("No data found in database")
            return None, None, None
        
        # Validate and fix sentiment scores
        df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
        
        # Add time-based features
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df['date'] = df['created_utc'].dt.date
        
        # Split multiple countries and leaders
        df['countries'] = df['country'].fillna('').str.split(',')
        df['leaders'] = df['leader'].fillna('').str.split(',')
        
        # Explode the dataframe
        df_countries = df.explode('countries').copy()
        df_leaders = df.explode('leaders').copy()
        
        # Clean data
        df_countries = df_countries[df_countries['countries'].str.strip() != '']
        df_leaders = df_leaders[df_leaders['leaders'].str.strip() != '']
        
        # Normalize country and leader names
        df_countries['countries'] = df_countries['countries'].str.strip().apply(normalize_country_name)
        df_leaders['leaders'] = df_leaders['leaders'].str.strip().apply(normalize_leader_name)
        
        return df, df_countries, df_leaders
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

def create_world_map(df_countries):
    """Create world map visualization"""
    if df_countries is None or df_countries.empty:
        return None
    
    try:
        # Агрегируем данные по нормализованным названиям стран
        country_stats = df_countries.groupby('countries').agg({
            'sentiment_score': ['mean', 'count'],
            'upvotes': 'mean'
        }).reset_index()
        
        country_stats.columns = ['country', 'avg_sentiment', 'post_count', 'avg_upvotes']
        
        # Добавляем специальную обработку для особых случаев
        def get_country_code(country_name):
            special_cases = {
                'USA': 'USA',
                'United Kingdom': 'GBR',
                'Russia': 'RUS',
                'European Union': 'EUR',
                'United Arab Emirates': 'ARE',
            }
            try:
                return special_cases.get(country_name) or pycountry.countries.search_fuzzy(country_name)[0].alpha_3
            except Exception:
                return None
        
        country_stats['iso_alpha'] = country_stats['country'].apply(get_country_code)
        country_stats = country_stats[country_stats['iso_alpha'].notna()]
        
        fig = px.choropleth(
            country_stats,
            locations='iso_alpha',
            color='avg_sentiment',
            hover_name='country',
            color_continuous_scale=[[0, '#d73027'], [0.5, '#ffffbf'], [1, '#4575b4']],
            range_color=[-1, 1],
            custom_data=['post_count', 'avg_sentiment']
        )
        
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                         "Sentiment: %{customdata[1]:.2f}<br>" +
                         "Posts: %{customdata[0]:,}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(
                showframe=True,
                framecolor='#666666',
                showcoastlines=True,
                coastlinecolor='#666666',
                showland=True,
                landcolor='#f0f0f0',
                showocean=True,
                oceancolor='#e6f3ff',
                showcountries=True,
                countrycolor='#666666',
                countrywidth=0.8,
                projection_type='equirectangular'
            ),
            coloraxis_colorbar=dict(
                title="Sentiment",
                ticktext=['Very Negative', 'Neutral', 'Very Positive'],
                tickvals=[-1, 0, 1],
                lenmode='fraction',
                len=0.75
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating world map: {e}")
        return None

def create_leader_stats(df_leaders):
    """Create leader statistics"""
    if df_leaders is None or df_leaders.empty:
        return pd.DataFrame()
    
    try:
        df_leaders['leaders'] = df_leaders['leaders'].apply(normalize_leader_name)
        
        leader_stats = df_leaders.groupby('leaders').agg({
            'sentiment_score': ['mean', 'count'],
            'upvotes': 'mean',
        }).reset_index()
        
        leader_stats.columns = ['leader', 'sentiment', 'mentions', 'avg_upvotes']
        leader_stats = leader_stats.sort_values('mentions', ascending=False)
        
        return leader_stats
    except Exception as e:
        logger.error(f"Error creating leader stats: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(
        page_title="Reddit Political Sentiment Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # Load data
    df, df_countries, df_leaders = load_data()
    
    if df is None:
        st.error("Error loading data from database")
        return
    
    # Header
    st.title("Reddit Political Sentiment Analysis")
    st.markdown("Real-time analysis of political discussions on Reddit")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Posts", f"{len(df):,}")
    with col2:
        st.metric("Countries Mentioned", f"{df_countries['countries'].nunique():,}")
    with col3:
        st.metric("Leaders Mentioned", f"{df_leaders['leaders'].nunique():,}")
    with col4:
        st.metric("Average Sentiment", f"{df['sentiment_score'].mean():.2f}")
    with col5:
        st.metric("Average Upvotes", f"{df['upvotes'].mean():.0f}")
    
    # World Map and Top Countries
    st.subheader("Global Sentiment Analysis")
    
    map_col, stats_col = st.columns([2, 1])
    
    with map_col:
        world_map = create_world_map(df_countries)
        if world_map:
            st.plotly_chart(world_map, use_container_width=True)
    
    with stats_col:
        country_stats = df_countries.groupby('countries').agg({
            'sentiment_score': ['mean', 'count']
        }).reset_index()
        country_stats.columns = ['Country', 'Sentiment', 'Posts']
        country_stats = country_stats[country_stats['Posts'] >= 10].sort_values('Sentiment')
        
        st.dataframe(
            country_stats.style.format({
                'Sentiment': '{:.2f}',
                'Posts': '{:,.0f}'
            }),
            hide_index=True
        )
    
    # Leader Analysis
    st.subheader("Leader Analysis")
    
    leader_stats = create_leader_stats(df_leaders)
    
    if not leader_stats.empty:
        # Top 3 Leaders Cards
        top_leaders = leader_stats.head(3)
        
        cols = st.columns(3)
        for i, (_, leader) in enumerate(top_leaders.iterrows()):
            with cols[i]:
                st.markdown(f"### #{i+1} {leader['leader']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mentions", f"{leader['mentions']:,}")
                with col2:
                    st.metric("Sentiment", f"{leader['sentiment']:.2f}")
                    st.metric("Avg. Upvotes", f"{leader['avg_upvotes']:.0f}")
        
        # Detailed Leader Rankings
        st.markdown("### Leader Rankings")
        
        # Создаем вкладки для разных рейтингов
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Overall Ranking",
            "💬 Most Discussed",
            "😊 Most Positive",
            "😠 Most Negative"
        ])
        
        with tab1:
            # Общий рейтинг
            overall_ranking = leader_stats.copy()
            overall_ranking['Rank'] = range(1, len(overall_ranking) + 1)
            overall_ranking = overall_ranking[[
                'Rank', 'leader', 'mentions',
                'sentiment', 'avg_upvotes'
            ]]
            overall_ranking.columns = [
                'Rank', 'Leader', 'Mentions',
                'Sentiment', 'Avg. Upvotes'
            ]
            
            st.dataframe(
                overall_ranking.style
                .format({
                    'Mentions': '{:,.0f}',
                    'Sentiment': '{:.2f}',
                    'Avg. Upvotes': '{:.0f}'
                })
                .background_gradient(
                    subset=['Sentiment'],
                    cmap='RdYlBu',
                    vmin=-1,
                    vmax=1
                ),
                hide_index=True,
                height=400
            )
        
        with tab2:
            # Рейтинг по количеству упоминаний
            mentions_ranking = leader_stats.sort_values('mentions', ascending=False).copy()
            mentions_ranking['Rank'] = range(1, len(mentions_ranking) + 1)
            mentions_ranking = mentions_ranking[[
                'Rank', 'leader', 'mentions',
                'sentiment', 'avg_upvotes'
            ]]
            mentions_ranking.columns = [
                'Rank', 'Leader', 'Mentions',
                'Sentiment', 'Avg. Upvotes'
            ]
            
            st.dataframe(
                mentions_ranking.style
                .format({
                    'Mentions': '{:,.0f}',
                    'Sentiment': '{:.2f}',
                    'Avg. Upvotes': '{:.0f}'
                })
                .background_gradient(
                    subset=['Sentiment'],
                    cmap='RdYlBu',
                    vmin=-1,
                    vmax=1
                ),
                hide_index=True,
                height=400
            )
        
        with tab3:
            # Рейтинг по позитивным настроениям
            positive_ranking = leader_stats[leader_stats['mentions'] >= 100].sort_values('sentiment', ascending=False).copy()
            positive_ranking['Rank'] = range(1, len(positive_ranking) + 1)
            positive_ranking = positive_ranking[[
                'Rank', 'leader', 'mentions',
                'sentiment', 'avg_upvotes'
            ]]
            positive_ranking.columns = [
                'Rank', 'Leader', 'Mentions',
                'Sentiment', 'Avg. Upvotes'
            ]
            
            st.dataframe(
                positive_ranking.style
                .format({
                    'Mentions': '{:,.0f}',
                    'Sentiment': '{:.2f}',
                    'Avg. Upvotes': '{:.0f}'
                })
                .background_gradient(
                    subset=['Sentiment'],
                    cmap='RdYlBu',
                    vmin=-1,
                    vmax=1
                ),
                hide_index=True,
                height=400
            )
        
        with tab4:
            # Рейтинг по негативным настроениям
            negative_ranking = leader_stats[leader_stats['mentions'] >= 100].sort_values('sentiment').copy()
            negative_ranking['Rank'] = range(1, len(negative_ranking) + 1)
            negative_ranking = negative_ranking[[
                'Rank', 'leader', 'mentions',
                'sentiment', 'avg_upvotes'
            ]]
            negative_ranking.columns = [
                'Rank', 'Leader', 'Mentions',
                'Sentiment', 'Avg. Upvotes'
            ]
            
            st.dataframe(
                negative_ranking.style
                .format({
                    'Mentions': '{:,.0f}',
                    'Sentiment': '{:.2f}',
                    'Avg. Upvotes': '{:.0f}'
                })
                .background_gradient(
                    subset=['Sentiment'],
                    cmap='RdYlBu',
                    vmin=-1,
                    vmax=1
                ),
                hide_index=True,
                height=400
            )
    
    # Time Analysis
    st.subheader("Sentiment Over Time")
    
    daily_stats = df.groupby('date').agg({
        'sentiment_score': ['mean', 'count']
    }).reset_index()
    daily_stats.columns = ['Date', 'Sentiment', 'Posts']
    
    fig = px.line(
        daily_stats,
        x='Date',
        y='Sentiment',
        custom_data=['Posts']
    )
    
    fig.update_traces(
        line_color='#4575b4',
        hovertemplate="Date: %{x}<br>" +
                     "Sentiment: %{y:.2f}<br>" +
                     "Posts: %{customdata[0]:,}<br>" +
                     "<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution
    st.subheader("Sentiment Distribution")
    
    fig = px.histogram(
        df,
        x='sentiment_score',
        nbins=30,
        color_discrete_sequence=['#4575b4'],
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Number of Posts",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Posts
    st.subheader("Top Posts")
    
    tab1, tab2, tab3 = st.tabs(["Most Positive", "Most Negative", "Most Upvoted"])
    
    with tab1:
        positive_posts = df.nlargest(5, 'sentiment_score')[
            ['text', 'sentiment_score', 'upvotes', 'created_utc']
        ]
        st.dataframe(
            positive_posts.style.format({
                'sentiment_score': '{:.2f}',
                'upvotes': '{:,.0f}',
                'created_utc': '{:%Y-%m-%d %H:%M}'
            }),
            hide_index=True
        )
    
    with tab2:
        negative_posts = df.nsmallest(5, 'sentiment_score')[
            ['text', 'sentiment_score', 'upvotes', 'created_utc']
        ]
        st.dataframe(
            negative_posts.style.format({
                'sentiment_score': '{:.2f}',
                'upvotes': '{:,.0f}',
                'created_utc': '{:%Y-%m-%d %H:%M}'
            }),
            hide_index=True
        )
    
    with tab3:
        top_posts = df.nlargest(5, 'upvotes')[
            ['text', 'sentiment_score', 'upvotes', 'created_utc']
        ]
        st.dataframe(
            top_posts.style.format({
                'sentiment_score': '{:.2f}',
                'upvotes': '{:,.0f}',
                'created_utc': '{:%Y-%m-%d %H:%M}'
            }),
            hide_index=True
        )

if __name__ == '__main__':
    main() 