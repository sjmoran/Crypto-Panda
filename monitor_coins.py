import os
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from coinpaprika import client as Coinpaprika
import time
from tabulate import tabulate
from dotenv import load_dotenv
import requests
import traceback
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai  # Assuming GPT-4 is accessible via OpenAI API
import json
import re
import math
import os
import logging 
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

surge_words = [
    "surge", "spike", "soar", "rocket", "skyrocket", "rally", "boom", "bullish", 
    "explosion", "rise", "uptrend", "bull run", "moon", "parabolic", "spurt", 
    "climb", "jump", "upswing", "gain", "increase", "growth", "rebound", 
    "breakout", "spurt", "pump", "fly", "explode", "shoot up", "hike", 
    "expand", "appreciate", "bull market", "peak", "momentum", "outperform", 
    "spike up", "ascend", "elevation", "expansion", "revive", "uprising", 
    "push up", "escalate", "rise sharply", "escalation", "recover", 
    "inflation", "strengthen", "gain strength", "intensify"
]

# Volume thresholds for liquidity risk
LOW_VOLUME_THRESHOLD_LARGE = 1_000_000  # Large-cap coins with daily volume under $1M
LOW_VOLUME_THRESHOLD_MID = 500_000  # Mid-cap coins with daily volume under $500k
LOW_VOLUME_THRESHOLD_SMALL = 100_000  # Small-cap coins with daily volume under $100k

# Configure logging
logging.basicConfig(
    filename='monitor_coins.log',  # Path to the log file
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Load environment variables from .env file
load_dotenv()

# Initialize the CoinPaprika client
client = Coinpaprika.Client(api_key=os.getenv('COIN_PAPRIKA_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Configuration
EMAIL_FROM = os.getenv('EMAIL_FROM') 
EMAIL_TO = os.getenv('EMAIL_TO') 
SMTP_SERVER = os.getenv('SMTP_SERVER')  
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_PORT = 587
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
RESULTS_FILE = "surging_coins.csv"
CRYPTO_NEWS_TICKERS = "tickers.csv"
FEAR_GREED_THRESHOLD = 60  # Fear and Greed index threshold
HIGH_VOLATILITY_THRESHOLD = 0.05  # 5% volatility is considered high
MEDIUM_VOLATILITY_THRESHOLD = 0.02  # 2% volatility is considered medium

# Flag to test with just Bitcoin
TEST_ONLY = False  # Set to False to monitor all coins
MAX_RETRIES = 2  # Maximum number of retries for API calls
BACKOFF_FACTOR = 2  # Factor by which the wait time increases after each failure

def api_call_with_retries(api_function, *args, **kwargs):
    """
    Calls an API function with retries in case of failure.

    Args:
        api_function (function): The API function to call.
        *args: Arguments to pass to the API function.
        **kwargs: Keyword arguments to pass to the API function.

    Returns:
        The result of the API function call.

    Raises:
        Exception: If the API function call fails after MAX_RETRIES retries.
    """
    retries = 0
    wait_time = 60  # Initial wait time (seconds)
    time.sleep(5)

    while retries < MAX_RETRIES:
        try:
            return api_function(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries < MAX_RETRIES:
                logging.debug(f"Error during API call: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= BACKOFF_FACTOR
            else:
                logging.debug(f"Max retries reached. Failed to complete API call.")
                raise

def fetch_fear_and_greed_index():
    """
    Fetches the current Fear and Greed Index value from the Alternative.me API.

    Returns:
        int: The current Fear and Greed Index value, or None if the API call fails.
    """
    try:
        response = api_call_with_retries(requests.get, 'https://api.alternative.me/fng/')
        data = response.json()
        return int(data['data'][0]['value'])
    except Exception as e:
        logging.debug(f"Error fetching Fear and Greed Index: {e}")
        return None

def fetch_historical_ticker_data(coin_id, start_date, end_date):
    """
    Fetches historical ticker data for the specified coin and date range.

    Parameters:
        coin_id (str): The CoinPaprika ID of the cryptocurrency.
        start_date (str): The start date of the period for which to fetch the ticker data (in YYYY-MM-DD format).
        end_date (str): The end date of the period for which to fetch the ticker data (in YYYY-MM-DD format).

    Returns:
        pandas.DataFrame: A DataFrame containing the historical ticker data, with columns 'date', 'price', 'coin_id', 'volume_24h', and 'market_cap'. 
                          If the API call fails, an empty DataFrame is returned.
    """
    try:
        historical_ticker = api_call_with_retries(client.historical, coin_id=coin_id, start=start_date, end=end_date, interval="1d", quote="usd")

        if isinstance(historical_ticker, list) and historical_ticker:
            df = pd.DataFrame(historical_ticker)

            if 'price' in df.columns and 'volume_24h' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                df['coin_id'] = coin_id
                df = df[['date', 'price', 'coin_id', 'volume_24h', 'market_cap']]
                df = df.sort_values(by='date')  # Sort the DataFrame by the 'date' column
                return df
            else:
                logging.debug(f"Missing expected columns in historical data for {coin_id}.")
                return pd.DataFrame()
        else:
            logging.debug(f"Unexpected data format returned for {coin_id}.")
            return pd.DataFrame()

    except Exception as e:
        logging.debug(f"Error fetching historical data for {coin_id}: {e}")
        return pd.DataFrame()

        
def filter_active_and_ranked_coins(coins, max_coins, rank_threshold=1000):
    """
    Filters the list of coins by rank, activity status, and new status, selecting up to max_coins.

    Parameters:
        coins (list): List of coins with rank, activity status, and new status information.
        max_coins (int): The maximum number of coins to return.
        rank_threshold (int): The maximum rank a coin must have to be included (e.g., rank <= 1000).

    Returns:
        list: Filtered list of coins that are active, not new, and ranked within the rank_threshold.
    """
    # Filter out coins that are not active, are new, or have a rank above the rank_threshold
    active_ranked_coins = [coin for coin in coins if coin.get('is_active', False) and not coin.get('is_new', True) and coin.get('rank', None) <= rank_threshold]

    # Limit the list to max_coins
    return active_ranked_coins[:max_coins]


def fetch_coin_events(coin_id):
    """
    Fetches events for a given cryptocurrency from the CoinPaprika API.

    Parameters:
        coin_id (str): The ID of the cryptocurrency.

    Returns:
        list: A list of recent events for the cryptocurrency, or an empty list if the API call fails.
    """
    try:
        events = api_call_with_retries(client.events, coin_id=coin_id)
        
        if not events:
            logging.debug(f"No events found for {coin_id}.")
            return []

        # Filter events to include only those from the past week and exclude future dates
        one_week_ago = datetime.now() - timedelta(days=7)
        recent_events = []

        for event in events:
            event_date = datetime.strptime(event['date'], "%Y-%m-%dT%H:%M:%SZ")
            if one_week_ago <= event_date <= datetime.now():
                recent_events.append(event)

        logging.debug(f"Events found for {coin_id}: {len(recent_events)} recent events")
        return recent_events

    except Exception as e:
        logging.debug(f"Error fetching events for {coin_id}: {e}")
        return []

def fetch_twitter_data(coin_id):
    """
    Fetches tweets for a given cryptocurrency from the CoinPaprika API.

    Parameters:
        coin_id (str): The ID of the cryptocurrency.

    Returns:
        pd.DataFrame: A pandas DataFrame containing tweets for the cryptocurrency from the past week, or an empty DataFrame if the API call fails or no tweets are found.
    """
    tweets = api_call_with_retries(client.twitter, coin_id)
    
    if not tweets:
        logging.debug(f"No tweets found for {coin_id}.")
        return pd.DataFrame()

    df = pd.DataFrame(tweets)

    if 'status' not in df.columns or 'date' not in df.columns:
        logging.debug(f"'status' or 'date' column not found in Twitter data for {coin_id}. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # Filter tweets from the past week
    one_week_ago = datetime.now() - timedelta(days=7)  # Naive datetime
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Convert to naive datetime
    df = df[df['date'] >= one_week_ago]

    if df.empty:
        logging.debug(f"No recent tweets (past week) found for {coin_id}.")
        return pd.DataFrame()

    logging.debug(f"Tweets found for {coin_id} in the past week: {len(df)} tweets")

    return df

def load_existing_results():
    """
    Loads existing results from a CSV file.

    Returns:
        pd.DataFrame: Existing results read from the CSV file, or an empty DataFrame if the file does not exist.
    """
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    else:
        return pd.DataFrame()

def save_result_to_csv(result):
    """
    Saves a single result as a row in a CSV file.

    The result will be appended to the existing file if it exists, or written to a new file if not.

    Parameters:
        result (dict): A dictionary containing at least the keys 'coin', 'market_cap', 'volume_24h', 'price_change_7d', and 'fear_greed_index'.
    """
    if not os.path.exists(RESULTS_FILE):
        pd.DataFrame([result]).to_csv(RESULTS_FILE, mode='w', header=True, index=False)
    else:
        pd.DataFrame([result]).to_csv(RESULTS_FILE, mode='a', header=False, index=False)

def classify_market_cap(market_cap):
    """
    Classify a market capitalization as "Large", "Mid", or "Small".

    Parameters:
        market_cap (int): The market capitalization of the cryptocurrency.

    Returns:
        str: The classification of the market capitalization.
    """
    if market_cap > 10_000_000_000:
        return "Large"
    elif market_cap > 1_000_000_000:
        return "Mid"
    else:
        return "Small"

def classify_volatility(volatility):
    """
    Classify a volatility value as "High", "Medium", or "Low".

    Parameters:
        volatility (float): The volatility of the cryptocurrency.

    Returns:
        str: The classification of the volatility.
    """
    if volatility > HIGH_VOLATILITY_THRESHOLD:
        return "High"
    elif volatility > MEDIUM_VOLATILITY_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def get_volume_thresholds(market_cap_class, volatility_class):
    """
    Returns the volume thresholds for a given market capitalization class and volatility class.

    Parameters:
        market_cap_class (str): The market capitalization class, one of "Large", "Mid", or "Small".
        volatility_class (str): The volatility class, one of "High", "Medium", or "Low".

    Returns:
        tuple: A tuple of six floats, representing the volume change thresholds for short-term, medium-term, and long-term periods, respectively.
    """
    thresholds = {
        ("Large", "High"): (2, 4, 1.5, 3, 1.2, 2),
        ("Large", "Medium"): (1.5, 3, 1.2, 2, 1.1, 1.5),
        ("Large", "Low"): (1.2, 2, 1.1, 1.5, 1, 1.2),
        ("Mid", "High"): (3, 6, 2, 4, 1.5, 2.5),
        ("Mid", "Medium"): (2, 4, 1.5, 3, 1.2, 2),
        ("Mid", "Low"): (1.5, 3, 1.2, 2, 1, 1.5),
        ("Small", "High"): (5, 10, 3, 6, 2, 4),
        ("Small", "Medium"): (3, 6, 2, 4, 1.5, 2.5),
        ("Small", "Low"): (2, 4, 1.5, 3, 1.2, 2)
    }
    return thresholds.get((market_cap_class, volatility_class), (2, 4, 1.5, 3, 1.2, 2))

def analyze_volume_change(volume_data, market_cap, volatility):
    """
    Analyze the volume changes of a cryptocurrency over three time periods.

    Parameters:
        volume_data (pd.DataFrame): Volume data for the cryptocurrency.
        market_cap (int): Market capitalization of the cryptocurrency.
        volatility (float): Volatility of the cryptocurrency.

    Returns:
        tuple: The volume change score and an explanation string detailing
               which periods had significant volume changes.
    """
    # Classify the market capitalization and volatility
    market_cap_class = classify_market_cap(market_cap)
    volatility_class = classify_volatility(volatility)

    # Get volume thresholds based on the market cap and volatility classification
    short_term_threshold, short_term_max, medium_term_threshold, medium_term_max, long_term_threshold, long_term_max = get_volume_thresholds(market_cap_class, volatility_class)

    # Analyze short-term, medium-term, and long-term volume changes
    short_term_change = calculate_volume_change(volume_data, period="short")
    medium_term_change = calculate_volume_change(volume_data, period="medium")
    long_term_change = calculate_volume_change(volume_data, period="long")

    volume_score = 0
    explanation_parts = []

    if short_term_change > short_term_threshold and short_term_change < short_term_max:
        volume_score += 1
        explanation_parts.append(f"Short-term volume change of {short_term_change*100:.2f}% exceeded the threshold of {short_term_threshold*100:.2f}%")
    
    if medium_term_change > medium_term_threshold and medium_term_change < medium_term_max:
        volume_score += 1
        explanation_parts.append(f"Medium-term volume change of {medium_term_change*100:.2f}% exceeded the threshold of {medium_term_threshold*100:.2f}%")
    
    if long_term_change > long_term_threshold and long_term_change < long_term_max:
        volume_score += 1
        explanation_parts.append(f"Long-term volume change of {long_term_change*100:.2f}% exceeded the threshold of {long_term_threshold*100:.2f}%")

    # Combine the explanation parts into a single string
    explanation = " | ".join(explanation_parts) if explanation_parts else "No significant volume changes detected."

    return volume_score, explanation


def has_consistent_weekly_growth(historical_df):
    # Calculate daily price changes
    """
    Returns True if the given historical data shows consistent weekly growth,
    which is defined as at least 4 out of the last 7 days having a positive
    price change.

    Parameters:
        historical_df (pd.DataFrame): A pandas DataFrame containing historical
            data for the cryptocurrency, with columns for 'date', 'price', and
            'volume_24h'.

    Returns:
        bool: True if the price has been growing for at least 4 of the last 7
            days, False otherwise.
    """
    historical_df['price_change'] = historical_df['price'].pct_change()

    # Filter the last 7 days
    last_week_df = historical_df.tail(7)
    
    # Count how many days had a positive price change
    rising_days = last_week_df[last_week_df['price_change'] > 0].shape[0]
    
    # Consider consistent growth if at least 4 out of 7 days were rising
    return rising_days >= 4

def has_sustained_volume_growth(historical_df):
    # Calculate daily volume changes
    """
    Returns True if the given historical data shows sustained volume growth,
    which is defined as at least 4 out of the last 7 days having a positive
    volume change.

    Parameters:
        historical_df (pd.DataFrame): A pandas DataFrame containing historical
            data for the cryptocurrency, with columns for 'date', 'price', and
            'volume_24h'.

    Returns:
        bool: True if the volume has been growing for at least 4 of the last 7
            days, False otherwise.
    """
    historical_df['volume_change'] = historical_df['volume_24h'].pct_change()

    # Filter the last 7 days
    last_week_df = historical_df.tail(7)
    
    # Count how many days had a positive volume change
    rising_volume_days = last_week_df[last_week_df['volume_change'] > 0].shape[0]
    
    # Consider sustained growth if at least 4 out of 7 days had rising volume
    return rising_volume_days >= 4


def compute_sentiment_for_coin(coin_name, news_data):
    """
    Computes the sentiment score for a given coin based on its news data.

    Parameters:
        coin_name (str): The name of the coin.
        news_data (list): A list of news items related to the coin, where each item is a dict with 'title' and 'description' keys.

    Returns:
        int: 1 if the average sentiment score is very positive (e.g., greater than 0.5), otherwise 0.
    """
    sentiments = []
    for news_item in news_data:
        description = news_item.get('description', '')
        
        # Ensure description is a string and is not empty
        if isinstance(description, str) and description.strip():
            sentiment_score = analyzer.polarity_scores(description)['compound']
            sentiments.append(sentiment_score)
    
    if sentiments:
        average_sentiment = sum(sentiments) / len(sentiments)
    else:
        average_sentiment = 0

    # Return 1 if the average sentiment is very positive (e.g., greater than 0.5), otherwise return 0
    return 1 if average_sentiment > 0.5 else 0

def score_surge_words(news_df, surge_words):
    """
    Analyze news articles and score the presence of surge words.

    Parameters:
        news_df (pd.DataFrame): A pandas DataFrame containing news articles.
        surge_words (list): A list of words indicating a surge in the market.

    Returns:
        tuple: A tuple containing the average surge score across all news articles 
               and a detailed explanation of which articles contributed to the score.
    """
    total_surge_score = 0
    news_count = 0
    explanation = []

    if not news_df.empty:
        for _, news_item in news_df.iterrows():
            description = news_item.get('description', '')

            # Ensure description is a string and is not None
            if isinstance(description, str) and description.strip():
                surge_score = 0
                article_explanation = []

                for word in surge_words:
                    # Fuzzy match each surge word with the description
                    match_score = fuzz.partial_ratio(word.lower(), description.lower())

                    # Add to surge_score based on the match score (scale 0 to 100)
                    if match_score > 75:  # Threshold for a significant match
                        surge_score += match_score / 100.0  # Normalize the score to 0-1 range
                        article_explanation.append(f"Matched word '{word}' with score {match_score}%")

                if surge_score > 0:
                    explanation.append(f"Article: '{news_item.get('title', '')}' contributed to surge score with details: {', '.join(article_explanation)}")

                total_surge_score += surge_score
                news_count += 1

    if news_count > 0:
        average_surge_score = total_surge_score / news_count
    else:
        average_surge_score = 0.0

    return int(math.ceil(average_surge_score)), explanation

def get_fuzzy_trending_score(coin_id, coin_name, trending_coins_scores):
    """
    Analyze trending coins scores and return the maximum score if a fuzzy match is found 
    between the coin ID or name and any of the trending coin tickers.

    Parameters:
        coin_id (str): The CoinPaprika ID of the cryptocurrency.
        coin_name (str): The full name of the cryptocurrency.
        trending_coins_scores (dict): A dictionary with tickers as keys and their respective scores.

    Returns:
        int: The maximum score if a fuzzy match is found, otherwise 0.
    """
    max_score = 0
    for ticker, score in trending_coins_scores.items():
        match_id = fuzz.partial_ratio(ticker.lower(), coin_id.lower())
        match_name = fuzz.partial_ratio(ticker.lower(), coin_name.lower())
        if match_id > 80 or match_name > 80:  # Threshold for considering a match
            max_score = max(max_score, score)
    return max_score


def classify_liquidity_risk(volume_24h, market_cap_class):
    """
    Classify liquidity risk based on trading volume.

    Parameters:
        volume_24h (float): The 24-hour trading volume of the cryptocurrency.
        market_cap_class (str): The market capitalization class, one of "Large", "Mid", or "Small".

    Returns:
        str: The classification of liquidity risk ('Low', 'Medium', 'High').
    """
    if market_cap_class == "Large":
        if volume_24h < LOW_VOLUME_THRESHOLD_LARGE:
            return "High"
        elif volume_24h < LOW_VOLUME_THRESHOLD_LARGE * 2:
            return "Medium"
        else:
            return "Low"
    elif market_cap_class == "Mid":
        if volume_24h < LOW_VOLUME_THRESHOLD_MID:
            return "High"
        elif volume_24h < LOW_VOLUME_THRESHOLD_MID * 2:
            return "Medium"
        else:
            return "Low"
    else:  # Small market cap
        if volume_24h < LOW_VOLUME_THRESHOLD_SMALL:
            return "High"
        elif volume_24h < LOW_VOLUME_THRESHOLD_SMALL * 2:
            return "Medium"
        else:
            return "Low"

def analyze_coin(coin_id, coin_name, end_date, news_df, digest_tickers, trending_coins_scores):
    """
    Analyzes a given cryptocurrency and returns a dictionary with various analysis scores, 
    including a score for whether the coin appears in the Sundown Digest and trending coins list.

    Parameters:
        coin_id (str): The CoinPaprika ID of the cryptocurrency.
        coin_name (str): The full name of the cryptocurrency.
        end_date (str): The end date of the period for which to fetch the historical ticker data (in YYYY-MM-DD format).
        news_df (pd.DataFrame): A DataFrame containing news articles related to the cryptocurrency.
        digest_tickers (list): A list of tickers extracted from the Sundown Digest.
        trending_coins_scores (dict): A dictionary with tickers as keys and their respective scores.

    Returns:
        dict: A dictionary with the analysis scores, cumulative score, and detailed explanation.
    """
    short_term_window = 7
    medium_term_window = 30
    long_term_window = 90
    
    start_date_short_term = (datetime.now() - timedelta(days=short_term_window)).strftime('%Y-%m-%d')
    start_date_medium_term = (datetime.now() - timedelta(days=medium_term_window)).strftime('%Y-%m-%d')
    start_date_long_term = (datetime.now() - timedelta(days=long_term_window)).strftime('%Y-%m-%d')

    historical_df_short_term = fetch_historical_ticker_data(coin_id, start_date_short_term, end_date)
    historical_df_medium_term = fetch_historical_ticker_data(coin_id, start_date_medium_term, end_date)
    historical_df_long_term = fetch_historical_ticker_data(coin_id, start_date_long_term, end_date)

    if 'price' not in historical_df_long_term.columns or historical_df_long_term.empty:
        logging.debug(f"No valid price data available for {coin_id}.")
        return {"coin_id": coin_id, "coin_name": coin_name, "explanation": f"No valid price data available for {coin_id}."}

    twitter_df = fetch_twitter_data(coin_id)
    tweet_score = 1 if not twitter_df.empty else 0

    volatility = historical_df_long_term['price'].pct_change().std()

    # Get price change score and detailed explanation
    price_change_score, price_change_explanation = analyze_price_change(historical_df_long_term['price'], historical_df_long_term['market_cap'].iloc[-1], volatility)

    volume_score, volume_explanation = analyze_volume_change(historical_df_long_term['volume_24h'], historical_df_long_term['market_cap'].iloc[-1], volatility)

    consistent_growth = has_consistent_weekly_growth(historical_df_short_term)
    consistent_growth_score = 1 if consistent_growth else 0

    sustained_volume_growth = has_sustained_volume_growth(historical_df_short_term)
    sustained_volume_growth_score = 1 if sustained_volume_growth else 0

    fear_and_greed_index = fetch_fear_and_greed_index()
    fear_and_greed_score = 1 if fear_and_greed_index is not None and fear_and_greed_index > FEAR_GREED_THRESHOLD else 0

    events = fetch_coin_events(coin_id)
    recent_events_count = sum(1 for event in events if datetime.strptime(event['date'], '%Y-%m-%d') <= datetime.now())
    event_score = 1 if recent_events_count > 0 else 0

    # Classify market cap
    market_cap_class = classify_market_cap(historical_df_long_term['market_cap'].iloc[-1])

    # Calculate liquidity risk based on the most recent 24-hour volume
    most_recent_volume_24h = historical_df_long_term['volume_24h'].iloc[-1]
    liquidity_risk = classify_liquidity_risk(most_recent_volume_24h, market_cap_class)

    # Integrate sentiment analysis
    if not news_df.empty:
        coin_news = news_df[news_df['coin'] == coin_name]
        sentiment_score = compute_sentiment_for_coin(coin_name, coin_news.to_dict('records'))

        # Integrate surge word analysis
        surge_score, surge_explanation = score_surge_words(coin_news, surge_words)
    else:
        sentiment_score = 0
        surge_score = 0
        surge_explanation = "No significant surge-related news detected."

    # Check if the coin is in the digest tickers
    digest_score = 1 if any(fuzz.partial_ratio(ticker.lower(), coin_id.lower()) > 80 or fuzz.partial_ratio(ticker.lower(), coin_name.lower()) > 80 for ticker in digest_tickers) else 0

    # Check if the coin is trending
    trending_score = get_fuzzy_trending_score(coin_id, coin_name, trending_coins_scores)

    # Calculate cumulative score and its percentage of the maximum score
    cumulative_score = (
        volume_score + tweet_score + consistent_growth_score + sustained_volume_growth_score + 
        fear_and_greed_score + event_score + price_change_score + sentiment_score + surge_score + digest_score + trending_score
    )

    # Maximum possible score (adjust as necessary)
    max_possible_score = 14 + trending_score  # Increased due to digest_score and trending_score

    # Calculate the cumulative score as a percentage
    cumulative_score_percentage = (cumulative_score / max_possible_score) * 100

    # Build the explanation string, including the detailed price change and surge word explanation
    explanation = f"{coin_name} ({coin_id}) analysis: "
    explanation += f"Liquidity Risk: {liquidity_risk}, "
    explanation += f"Price Change Score: {'Significant' if price_change_score else 'No significant change'} ({price_change_explanation}), "
    explanation += f"Volume Change Score: {'Significant' if volume_score else 'No significant change'} ({volume_explanation}), "
    explanation += f"Tweets: {'Yes' if tweet_score else 'None'}, "
    explanation += f"Consistent Price Growth: {'Yes' if consistent_growth_score else 'No'}, "
    explanation += f"Sustained Volume Growth: {'Yes' if sustained_volume_growth_score else 'No'}, "
    explanation += f"Fear and Greed Index: {fear_and_greed_index if fear_and_greed_index is not None else 'N/A'}, "
    explanation += f"Recent Events: {recent_events_count}, "
    explanation += f"Sentiment Score: {sentiment_score}, "
    explanation += f"Surge Keywords Score: {surge_score} ({surge_explanation}), "
    explanation += f"News Digest Score: {digest_score}, "
    explanation += f"Trending Score: {trending_score}, "
    explanation += f"Cumulative Surge Score: {cumulative_score} ({cumulative_score_percentage:.2f}%)"

    return {
        "coin_id": coin_id,
        "coin_name": coin_name,
        "price_change_score": f"{price_change_score}",
        "volume_change_score": f"{volume_score}",
        "tweets": len(twitter_df) if tweet_score else "None",
        "consistent_growth": "Yes" if consistent_growth_score else "No",
        "sustained_volume_growth": "Yes" if sustained_volume_growth_score else "No",
        "fear_and_greed_index": fear_and_greed_index if fear_and_greed_index is not None else "N/A",
        "events": recent_events_count,
        "sentiment_score": sentiment_score,
        "surging_keywords_score": surge_score,
        "news_digest_score": digest_score,
        "trending_score": trending_score,
        "liquidity_risk": liquidity_risk,  # Added liquidity risk
        "cumulative_score": cumulative_score,
        "cumulative_score_percentage": round(cumulative_score_percentage, 2),  # Rounded to 2 decimal places
        "explanation": explanation
    }


def load_tickers(file_path):
    """
    Loads a CSV file containing coin names and tickers, and returns a dictionary mapping
    the coin names to their tickers.

    Parameters:
        file_path (str): The path to the CSV file to load.

    Returns:
        dict: A dictionary mapping coin names to their tickers.
    """
    tickers_df = pd.read_csv(file_path)
    # Create a dictionary mapping the coin names to their tickers
    tickers_dict = pd.Series(tickers_df['Ticker'].values, index=tickers_df['Name']).to_dict()
    return tickers_dict

# Fetch news using the tickers from the loaded CSV
def fetch_news_for_past_week(tickers_dict):
    """
    Fetches news for each coin in the given tickers dictionary for the past week.

    Parameters:
        tickers_dict (dict): A dictionary mapping the coin names to their tickers.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the news articles fetched from the API, with columns 'coin', 'date', 'title', 'description', 'url', and 'source'.
    """
    all_news = []

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    for coin_name, coin_ticker in tickers_dict.items():
        logging.debug(f"Fetching news for {coin_name} ({coin_ticker})...")

        formatted_date = end_date.strftime('%Y%m%d')
        week_start = (end_date - timedelta(days=3)).strftime('%m%d%Y')
        week_end = end_date.strftime('%m%d%Y')
        date_str = f"{week_start}-{week_end}"

        url = f"https://cryptonews-api.com/api/v1?tickers={coin_ticker}&items=1&date={date_str}&sortby=rank&token={os.getenv('CRYPTO_NEWS_API_KEY')}"
        response = requests.get(url)
        if response.status_code == 200:
                news_data = response.json()
                if 'data' in news_data and news_data['data']:
                    for article in news_data['data']:
                        all_news.append({
                            "coin": coin_name,
                            "date": formatted_date,  # Log the specific day
                            "title": article["title"],
                            "description": article.get("text", ""),
                            "url": article["news_url"],
                            "source": article["source_name"]
                        })
                else:
                    logging.debug(f"No news for {coin_name} between {week_start} and {week_end}.")
        else:
                logging.debug(f"Failed to fetch news for {coin_name} between {week_start} and {week_end}. Status Code: {response.status_code}")
            
        time.sleep(1)
        end_date -= timedelta(days=1)

    return pd.DataFrame(all_news)

def save_result_to_csv(result):
    """
    Saves a single result as a row in a CSV file.

    The result will be appended to the existing file if it exists, or written to a new file if not.

    Parameters:
        result (dict): A dictionary containing at least the keys 'coin', 'market_cap', 'volume_24h', 'price_change_7d', and 'fear_greed_index'.
    """
    if not os.path.exists(RESULTS_FILE):
        # If the file doesn't exist, create it with headers
        pd.DataFrame([result]).to_csv(RESULTS_FILE, mode='w', header=True, index=False)
    else:
        # If the file exists, append to it without writing headers again
        pd.DataFrame([result]).to_csv(RESULTS_FILE, mode='a', header=False, index=False)


def gpt4o_summarize_digest_and_extract_tickers(digest_text):
    """
    Uses GPT-4 to summarize the Sundown Digest and extract key points related to potential surges in coin value.

    Args:
        digest_text (str): The concatenated text from all digest entries.

    Returns:
        dict: A dictionary containing a summary focused on surge-causing news and a list of extracted tickers.
    """
    prompt = f"""
    Analyze the following digest entries and provide the following:
    1) A concise summary in bullet points (no more than 250 words) of key news items likely to cause surges in the value of the mentioned coins. 
    2) List the relevant cryptocurrency tickers beside each news item.

    Text:
    {digest_text}

    Respond **only** in JSON format with 'surge_summary' and 'tickers' as keys.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        # Extract the content of the response
        response_content = response.choices[0].message['content'].strip()
        # Use a regular expression to extract JSON from the response content
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                analysis = json.loads(json_str)
                return analysis
            except json.JSONDecodeError:
                logging.debug(f"Failed to decode JSON: {json_str}")
                return {"surge_summary": "", "tickers": []}
        else:
            logging.debug(f"No JSON found in the response: {response_content}")
            return {"surge_summary": "", "tickers": []}
        
    except openai.error.RateLimitError as e:
        logging.debug(f"Rate limit reached: {e}. Waiting for 60 seconds before retrying...")
        time.sleep(60)  # Wait before retrying
        return gpt4o_summarize_digest_and_extract_tickers(digest_text)  # Retry the request

    except Exception as e:
        logging.debug(f"An error occurred while summarizing the digest and extracting tickers: {e}")
        return {"surge_summary": "", "tickers": []}


def summarize_sundown_digest(digest):
    """
    Summarizes the Sundown Digest content from the last three days, including sentiment detection,
    coin ticker extraction, and a summary focused on news likely to cause surges in coin value.

    Args:
        digest (list): List of Sundown Digest entries.

    Returns:
        dict: A dictionary containing key points of news that may cause surges and relevant tickers.
    """
    # Get the current date and calculate the date three days ago
    current_date = datetime.now()
    three_days_ago = current_date - timedelta(days=3)

    digest_texts = []
    
    for entry in digest:
        # Parse the entry's date
        entry_date = datetime.strptime(entry['date'], '%Y-%m-%dT%H:%M:%S.%fZ')

        # Filter out entries older than three days
        if entry_date < three_days_ago:
            continue

        digest_texts.append(entry['text'])

    # Concatenate all digest texts into a single string
    combined_digest_text = " ".join(digest_texts)

    # Use GPT-4 to analyze and summarize the combined digest text
    summary_and_tickers = gpt4o_summarize_digest_and_extract_tickers(combined_digest_text)
    return summary_and_tickers


def get_sundown_digest():
    """
    Fetches the Sundown Digest from CryptoNews API.

    Returns:
        dict: A dictionary containing the Sundown Digest data.
    """
    url = f"https://cryptonews-api.com/api/v1/sundown-digest?page=1&token={os.getenv('CRYPTO_NEWS_API_KEY')}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        sundown_digest = response.json()
        return sundown_digest.get('data', [])
    except requests.exceptions.HTTPError as http_err:
        logging.debug(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logging.debug(f"An error occurred: {err}")
    return []

def monitor_coins_and_send_report():
    """
    Main entry point to monitor the specified coins, fetch news, analyze sentiment,
    and send a report with the results.

    If TEST_ONLY is set to True, only a few coins are processed and the results are
    saved to a file. Otherwise, all coins are processed and the results are sent
    via email.
    """
    if TEST_ONLY:

        # Test only mode: we only process a predefined list of coins
        existing_results = pd.DataFrame([])
        # A predefined list of 10-20 coins to monitor for testing purposes
        coins_to_monitor = [
            # Large Cap Cryptocurrencies
            {"id": "nmr-numeraire", "name": "Numeraire"},
            {"id": "btc-bitcoin", "name": "Bitcoin"},
            {"id": "eth-ethereum", "name": "Ethereum"},
            {"id": "bnb-binancecoin", "name": "Binance Coin"},
            {"id": "ada-cardano", "name": "Cardano"},
            {"id": "xrp-ripple", "name": "Ripple"},
            
            # Mid Cap Cryptocurrencies
            {"id": "algo-algorand", "name": "Algorand"},
            {"id": "ftm-fantom", "name": "Fantom"},
            {"id": "near-near-protocol", "name": "Near Protocol"},
            {"id": "mana-decentraland", "name": "Decentraland"},
            {"id": "grt-the-graph", "name": "The Graph"},
            {"id": "icp-internet-computer", "name": "Internet Computer"},
            {"id": "hbar-hedera", "name": "Hedera"},
            {"id": "sand-the-sandbox", "name": "The Sandbox"},
            {"id": "enj-enjin-coin", "name": "Enjin Coin"},
            
            # Small Cap Cryptocurrencies
            {"id": "rune-thorchain", "name": "THORChain"},
            {"id": "rndr-render-token", "name": "Render Token"},
            {"id": "ogn-origin-protocol", "name": "Origin Protocol"},
            {"id": "ctsi-cartesi", "name": "Cartesi"},
            {"id": "holo-holo", "name": "Holo"},
            {"id": "storj-storj", "name": "Storj"},
        ]

    else:
        # Normal mode: retrieve coins from CoinPaprika API
        existing_results = load_existing_results()
        coins_to_monitor = api_call_with_retries(client.coins)

        # Report the number of coins retrieved
        num_coins_retrieved = len(coins_to_monitor)
        logging.debug(f"Number of coins retrieved: {num_coins_retrieved}")
        print(f"Number of coins retrieved: {num_coins_retrieved}")

        # Filter coins based on rank and active status, limit to 1000 coins
        coins_to_monitor = filter_active_and_ranked_coins(coins_to_monitor, 71909)

    logging.debug(f"Number of active and ranked coins selected: {len(coins_to_monitor)}")
    print(f"Number of active and ranked coins selected: {len(coins_to_monitor)}")

    end_date = datetime.now().strftime('%Y-%m-%d')

    report_entries = []

    tickers_dict = load_tickers(CRYPTO_NEWS_TICKERS)

    # Fetch and summarize the Sundown Digest
    sundown_digest = get_sundown_digest()
    digest_summary = summarize_sundown_digest(sundown_digest)
    digest_tickers = digest_summary['tickers']

    # Fetch Trending Coins data once
    trending_coins_scores = fetch_trending_coins_scores()

    for coin in coins_to_monitor:
        try:
            coin_id = coin['id']
            coin_name = coin['name']

            if not existing_results.empty and coin_id in existing_results['coin_id'].values:
                logging.debug(f"Skipping already processed coin: {coin_id}")
                continue

            # Fetch news directly for analysis
            coins_dict = {coin_name: tickers_dict.get(coin_name, '').upper()}
            news_df = fetch_news_for_past_week(coins_dict)

            # Analyze coin and save the result
            result = analyze_coin(coin_id, coin_name, end_date, news_df, digest_tickers, trending_coins_scores)
            logging.debug(f"Result for {coin_name}: {result}")

            save_result_to_csv(result)
            report_entries.append(result)

            time.sleep(20)

        except Exception as e:
            logging.debug(f"An error occurred while processing {coin_name} ({coin_id}): {e}")
            logging.debug(traceback.format_exc())
            continue

    # Final report generation...

    # Ensure all cumulative_score values are numeric (default to 0 if None)
    for entry in report_entries:
        if entry['cumulative_score'] is None:
            entry['cumulative_score'] = 0

    # Final report generation if everything else succeeds
    if report_entries:
        report_entries = sorted(report_entries, key=lambda x: x.get('cumulative_score', 0), reverse=True)
        print_command_line_report(report_entries, digest_summary)
        html_report = generate_html_report(report_entries, digest_summary)  # Pass digest_summary here
        send_email_with_report(html_report)
        
        # Delete the surging_coins.csv file after sending the email
        if os.path.exists(RESULTS_FILE):
            try:
                os.remove(RESULTS_FILE)
                logging.debug(f"{RESULTS_FILE} has been deleted successfully.")
            except Exception as e:
                logging.debug(f"Failed to delete {RESULTS_FILE}: {e}")
    else:
        logging.debug("No valid entries to report.")


def normalize_score(raw_score, min_score, max_score, range=3):
    """
    Normalize a score to be between 0 and 3 and return the nearest integer.

    Args:
        raw_score (float): The original score.
        min_score (float): The minimum score in the data.
        max_score (float): The maximum score in the data.

    Returns:
        int: The normalized score between 0 and 3, rounded to the nearest integer.
    """
    # Handle the edge case where all scores might be the same
    if max_score == min_score:
        return 2  # Midpoint value since all scores are the same

    # Normalize to a 0-1 range
    normalized = (raw_score - min_score) / (max_score - min_score)
    # Scale to a 0-3 range and round to the nearest integer
    return round(normalized * range)


def fetch_trending_coins_scores():
    """
    Fetches trending coins from CryptoNews API and calculates their sentiment scores.
    Normalizes the scores between 0 and 3 and returns a dictionary with the trending coins and their normalized scores.

    Returns:
        dict: A dictionary with the trending coins as keys and their normalized scores as values.
    """
    url = f"https://cryptonews-api.com/api/v1/top-mention?&date=last7days&token={os.getenv('CRYPTO_NEWS_API_KEY')}"
    response = requests.get(url)
    trending_data = response.json()['data']['all']

    trending_coins_scores = {}
    raw_scores = {}

    # Calculate raw scores
    for item in trending_data:
        ticker = item['ticker'].lower()
        sentiment_score = item['sentiment_score']
        total_mentions = item['total_mentions']
        raw_score = sentiment_score * total_mentions
        raw_scores[ticker] = raw_score

    # Determine min and max raw scores for normalization
    min_raw_score = min(raw_scores.values())
    max_raw_score = max(raw_scores.values())

    # Normalize scores between 0 and 3
    for ticker, raw_score in raw_scores.items():
        trending_coins_scores[ticker] = normalize_score(raw_score, min_raw_score, max_raw_score)

    return trending_coins_scores


def send_failure_email():
    """
    Sends an email with the current results when the script encounters an error.

    This function reads the current results from the file specified by RESULTS_FILE,
    generates an HTML email with the results and sends it to the recipient specified
    by EMAIL_TO.

    If the file does not exist, the email will state that no data is available.

    The email is sent using the SMTP server specified by SMTP_SERVER and the
    credentials specified by SMTP_USERNAME and SMTP_PASSWORD.

    If sending the email fails, an error message is logging.debuged.
    """
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as file:
            file_contents = file.read()
    else:
        file_contents = "No data available, as the results file was not created."

    # HTML content with inline CSS for the failure email
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                color: #333;
            }}
            h2 {{
                color: #c0392b;
            }}
            p {{
                font-size: 14px;
                color: #555;
            }}
            .content {{
                background-color: #f9f9f9;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .content pre {{
                background-color: #f4f4f4;
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <h2>Failure in Weekly Coin Analysis Script</h2>
        <p>The script encountered an error. Below are the current results:</p>
        <div class="content">
            <pre>{file_contents}</pre>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Failure in Weekly Coin Analysis Script"
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    part = MIMEText(html_content, 'html')
    msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        logging.debug("Failure email sent successfully.")
    except Exception as e:
        logging.debug(f"Failed to send email: {e}")


def get_price_change_thresholds(market_cap_class, volatility_class):
    """
    Returns the price change thresholds for a given market capitalization class and volatility class.

    Parameters:
        market_cap_class (str): The market capitalization class, one of "Large", "Mid", or "Small".
        volatility_class (str): The volatility class, one of "High", "Medium", or "Low".

    Returns:
        tuple: A tuple of three floats, representing the price change thresholds for short-term, medium-term, and long-term periods, respectively.
    """
    thresholds = {
        ("Large", "High"): (0.03, 0.02, 0.01),
        ("Large", "Low"): (0.015, 0.01, 0.005),
        ("Mid", "High"): (0.05, 0.03, 0.02),
        ("Mid", "Medium"): (0.03, 0.02, 0.015),
        ("Mid", "Low"): (0.02, 0.015, 0.01),
        ("Small", "High"): (0.07, 0.05, 0.03),
        ("Small", "Medium"): (0.05, 0.03, 0.02),
        ("Small", "Low"): (0.03, 0.02, 0.015)
    }
    return thresholds.get((market_cap_class, volatility_class), (0.03, 0.02, 0.01))

def analyze_price_change(price_data, market_cap, volatility):
    """
    Analyze the price changes of a cryptocurrency over three time periods.

    Parameters:
        price_data (pd.DataFrame): Price data for the cryptocurrency.
        market_cap (int): Market capitalization of the cryptocurrency.
        volatility (float): Volatility of the cryptocurrency.

    Returns:
        tuple: The price change score and an explanation string detailing
               which periods had significant price changes.
    """
    market_cap_class = classify_market_cap(market_cap)
    volatility_class = classify_volatility(volatility)

    # Unpack only the lower threshold values
    short_term_threshold, medium_term_threshold, long_term_threshold = get_price_change_thresholds(market_cap_class, volatility_class)

    # Analyze short-term, medium-term, and long-term price changes
    short_term_change = calculate_price_change(price_data, period="short")
    medium_term_change = calculate_price_change(price_data, period="medium")
    long_term_change = calculate_price_change(price_data, period="long")

    price_change_score = 0
    explanation_parts = []

    if short_term_change > short_term_threshold:
        price_change_score += 1
        explanation_parts.append(f"Short-term change of {short_term_change*100:.2f}% exceeded the threshold of {short_term_threshold*100:.2f}%")
    
    if medium_term_change > medium_term_threshold:
        price_change_score += 1
        explanation_parts.append(f"Medium-term change of {medium_term_change*100:.2f}% exceeded the threshold of {medium_term_threshold*100:.2f}%")
    
    if long_term_change > long_term_threshold:
        price_change_score += 1
        explanation_parts.append(f"Long-term change of {long_term_change*100:.2f}% exceeded the threshold of {long_term_threshold*100:.2f}%")

    explanation = " | ".join(explanation_parts) if explanation_parts else "No significant price changes detected."

    return price_change_score, explanation


def calculate_price_change(price_data, period="short", span=7):
    """
    Calculate the percentage change in price over a given period with optional smoothing using Exponential Moving Average (EMA).

    Parameters:
        price_data (pd.DataFrame): Price data for the cryptocurrency with a single column.
        period (str, optional): The time period to calculate the price change for. Options are "short" (7 days), "medium" (30 days), and "long" (90 days). Defaults to "short".
        span (int, optional): The span for the EMA smoothing. Defaults to 7.

    Returns:
        float: The percentage change in price over the specified period, smoothed by EMA.
    """
    # Apply EMA
    smoothed_data = price_data.ewm(span=span, adjust=False).mean()

    if period == "short":
        period_data = smoothed_data.tail(7)  # Last 7 days
    elif period == "medium":
        period_data = smoothed_data.tail(30)  # Last 30 days
    else:  # long term
        period_data = smoothed_data.tail(90)  # Last 90 days

    start_price = period_data.iloc[0]  # Extract the first value from the single column
    end_price = period_data.iloc[-1]  # Extract the last value from the single column

    if start_price == 0:
        return None  # or some other appropriate value or handling mechanism
    return (end_price - start_price) / start_price


def calculate_volume_change(volume_data, period="short", span=7):
    """
    Calculate the percentage change in volume over a given period with optional smoothing using Exponential Moving Average (EMA).

    Parameters:
        volume_data (pd.DataFrame): Volume data for the cryptocurrency with a single column.
        period (str, optional): The time period to calculate the volume change for. Options are "short" (7 days), "medium" (30 days), and "long" (90 days). Defaults to "short".
        span (int, optional): The span for the EMA smoothing. Defaults to 7.

    Returns:
        float: The percentage change in volume over the specified period, smoothed by EMA.
    """
    # Apply EMA
    smoothed_data = volume_data.ewm(span=span, adjust=False).mean()

    if period == "short":
        period_data = smoothed_data.tail(7)  # Last 7 days
    elif period == "medium":
        period_data = smoothed_data.tail(30)  # Last 30 days
    else:  # long term
        period_data = smoothed_data.tail(90)  # Last 90 days

    start_volume = period_data.iloc[0]  # Extract the first value from the single column
    end_volume = period_data.iloc[-1]  # Extract the last value from the single column

    if start_volume == 0:
        return None  # or some other appropriate value or handling mechanism
    return (end_volume - start_volume) / start_volume

def print_command_line_report(report_entries):
    """
    Prints a command-line report of the daily coin analysis.

    Parameters
    ----------
    report_entries : list
        A list of dictionaries, each containing the analysis results for a single coin.

    Returns
    -------
    None
    """
    df = pd.DataFrame(report_entries)
    logging.debug("\nWeekly Report: Coin Analysis")
    logging.debug(tabulate(df, headers="keys", tablefmt="grid"))
    logging.debug(f"\nReport generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def print_command_line_report(report_entries, digest_summary):
    """
    Prints a command-line report of the daily coin analysis, with the digest summary
    printed separately before the table.

    Parameters:
    report_entries (list): A list of dictionaries, each containing the analysis results for a single coin.
    digest_summary (dict): A dictionary containing the summarized Sundown Digest.
    
    Returns:
    None
    """
    # logging.debug the Sundown Digest summary first
    logging.debug("\nSundown Digest Summary:")

    logging.debug("\nTickers Mentioned:")
    logging.debug(", ".join(digest_summary['tickers']))

    logging.debug("\nNews Sumnmary:")
    for surge in digest_summary['surge_summary']:
        logging.debug(f"- {surge}")

    # Then logging.debug the table of analyzed coins
    df = pd.DataFrame(report_entries)
    logging.debug("\nWeekly Report: Coin Analysis")
    logging.debug(tabulate(df, headers="keys", tablefmt="grid"))
    logging.debug(f"\nReport generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def generate_html_report(report_entries, digest_summary):
    """
    Generates an HTML report from the report entries and includes the Sundown Digest summary.

    Args:
        report_entries (list): List of report entries to include in the report.
        digest_summary (dict): Summary of the Sundown Digest to include at the top.

    Returns:
        str: HTML content of the report.
    """
    df = pd.DataFrame(report_entries)

    # Create a new column with links to CoinPaprika for each coin
    df['coin_link'] = df['coin_id'].apply(lambda coin_id: f'<a href="https://coinpaprika.com/coin/{coin_id}/">View on CoinPaprika</a>')

    # Sort the DataFrame by cumulative score in descending order
    df = df.sort_values(by="cumulative_score", ascending=False)

    # Format the column headers to proper words without underscores
    df.columns = df.columns.str.replace('_', ' '). str.title()

    # Rename the 'Coin Link' column to match the formatting
    df.rename(columns={"Coin Link": "Link To Coin"}, inplace=True)

    # Check if 'Explanation' column exists before applying the lambda function
    if 'Explanation' in df.columns:
        df['Explanation'] = df['Explanation'].apply(lambda x: x.replace(" | ", "<br>"))

    # Create the HTML table with enhanced styling and include the link
    html_table = df.to_html(index=False, escape=False, classes='table table-striped', table_id="crypto-table")

    # Create the Sundown Digest section with improved styling
    digest_html = f"""
    <div style="background-color:#f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #4CAF50;">Sundown News Digest Summary</h3>
        <p><strong>Tickers:</strong> {', '.join(digest_summary['tickers'])}</p>
        <p><strong>News Summary:</strong></p>
        <ul style="padding-left: 20px;">
            {''.join(f'<li>{item}</li>' for item in digest_summary['surge_summary'])}
        </ul>
    </div>
    """

    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                color: #333;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
            }}
            h2 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }}
            #crypto-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            #crypto-table th, #crypto-table td {{
                padding: 10px 15px;
                border: 1px solid #ddd;
                text-align: left;
                vertical-align: top;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            #crypto-table th {{
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }}
            #crypto-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            #crypto-table tr:hover {{
                background-color: #f1f1f1;
            }}
        </style>
    </head>
    <body>
        <h2>Weekly Report: Coin Analysis</h2>
        {digest_html}
        {html_table}
        <p style="text-align: center; color: #777;">Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """
    return html_content


def send_email_with_report(html_content):
    """
    Sends an email with the given HTML content to the recipient specified by EMAIL_TO.

    This function uses the Brevo SMTP server to send the email.

    Parameters:
        html_content (str): The HTML content of the email.

    Returns:
        None
    """
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Weekly Report: Coin Analysis"
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    part = MIMEText(html_content, 'html')
    msg.attach(part)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

if __name__ == "__main__":
    monitor_coins_and_send_report()
