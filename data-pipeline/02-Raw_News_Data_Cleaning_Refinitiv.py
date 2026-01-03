# !pip install clean-text
# !pip install unidecode
# !pip install Levenshtein

import re
import pandas as pd
import yfinance as yf
from cleantext import clean
from Levenshtein import ratio
from datetime import datetime, timedelta
import os
import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Project root directory
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "01_raw")

def extract_update_number(headline):
    """
    Extracts the update number from the beginning of a news headline.
    
    Parameters:
        headline (str): The headline string to be processed.

    Returns:
        int: The extracted update number, or 0 if not present.
    """
    match = re.match(r'UPDATE (\d+)', headline)
    return int(match.group(1)) if match else 0

def create_new_headline(row):
    """
    Creates a new headline by removing the update number if it exists.
    
    Parameters:
        row (pd.Series): A row of DataFrame.

    Returns:
        str: The processed headline.
    """
    if pd.notnull(row['update_number']):
        return re.sub(r'UPDATE \d+-', '', row['headline'])
    else:
        return row['headline']

def clean_news(df):
    """
    Cleans the 'body' column of the DataFrame by trimming text before '(Reuters)'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    for index, row in df.iterrows():
        if '(Reuters)' in row['body']:
            position = row['body'].find('(Reuters)')
            df.at[index, 'body'] = row['body'][position:]
    return df

def remove_spaces(df, column_names):
    """
    Removes spaces from specified columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.
        column_names (list of str): List of column names to process.

    Returns:
        pd.DataFrame: The DataFrame with spaces removed from specified columns.
    """
    for column in column_names:
        temp_column = f'temp_{column}'
        df[temp_column] = df[column].str.replace(' ', '', regex=True)
    return df

def replace_column_values(df, column_name, new_value):
    """
    Replace all values in a specified column of a DataFrame with a new value.

    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.
        column_name (str): Name of the column in the DataFrame.
        new_value (str): The new value to replace with.

    Returns:
        pd.DataFrame: The DataFrame with updated values.
    """
    if column_name in df.columns:
        df[column_name] = new_value
    else:
        print(f"Column '{column_name}' not found in DataFrame.")
    return df

def calculate_date(row):
    """
    Calculates the date based on the hour of the day.

    Parameters:
        row (pd.Timestamp): A timestamp object.

    Returns:
        datetime.date: The calculated date.
    """
    if row.hour >= 16:
        return (row + timedelta(days=1)).date()
    else:
        return row.date()

def clean_text(text):
    """
    Cleans a text string using the 'clean' function from the cleantext library.

    Parameters:
        text (str): The text string to be cleaned.

    Returns:
        str: The cleaned text string, or empty string if input is None/NaN.
    """
    # Handle None, NaN, or empty values
    if pd.isna(text) or text is None or text == '':
        return ''

    # Convert to string if not already
    text = str(text)

    # Use clean-text library with correct API
    # clean_all=False to control what gets cleaned
    # Keep numbers, digits, currency symbols, and punctuation (important for financial text)
    return clean(text,
                 clean_all=False,
                 extra_spaces=True,
                 stemming=False,
                 stopwords=False,
                 lowercase=True,
                 numbers=False,      # Keep numbers
                 punct=False,        # Keep punctuation
                 stp_lang='english'
                )

def drop_similar_records(df, column_name, r):
    """
    Drops records in a DataFrame that have a high similarity in a specified column.

    The function uses the Levenshtein distance to compute the similarity between 
    consecutive rows in the specified column. Rows are marked for dropping if the 
    similarity ratio is greater than the threshold 'r'.

    Input:
        df (pandas.DataFrame): The DataFrame to process.
        column_name (str): The name of the column to check for similarity.
        r (float): The threshold ratio for similarity. Rows with similarity above this 
                   value are marked for dropping.

    Output:
        pandas.DataFrame: A DataFrame with similar rows removed based on the specified 
                          column and similarity threshold.

    Notes:
        - The column specified by 'column_name' is converted to string type to ensure 
          proper comparison.
        - The function assumes that 'df' is a pandas DataFrame and 'column_name' exists in it.
        - The Levenshtein ratio ranges from 0 to 1, where 1 indicates identical strings.
    """

    # Ensure the column is of string type
    df[column_name] = df[column_name].astype(str)
    
    # Create a new column to mark rows to be dropped
    df['drop_row'] = False
    
    # Iterate over the DataFrame rows
    for i in range(1, len(df)):
        # Calculate the similarity ratio using Levenshtein distance
        similarity = ratio(df.at[i, column_name], df.at[i-1, column_name])
        # Mark the row for dropping if the similarity is above r
        if similarity > r:
            df.at[i, 'drop_row'] = True
    
    # Drop the marked rows
    df = df[~df['drop_row']].drop('drop_row', axis=1)
    
    return df


def adjust_trading_days(start_day, end_day, ticker, df):
    """
    Adjusts the dates in a DataFrame to the nearest following trading days
    based on stock data from Yahoo Finance.

    This function takes a DataFrame and modifies its date column to ensure that
    each date falls on a trading day. Non-trading days are adjusted to the next
    trading day. Trading days are determined based on the stock data for the
    specified ticker within the given date range.

    Input:
        start_day (str): The start date for the trading day range in 'YYYY-MM-DD' format.
        end_day (str): The end date for the trading day range in 'YYYY-MM-DD' format.
        ticker (str): The stock ticker symbol used to fetch trading day data from Yahoo Finance.
        df (pandas.DataFrame): The DataFrame containing a 'date' column to be adjusted.

    Output:
        pandas.DataFrame: The input DataFrame with its 'date' column adjusted to
                          the nearest following trading days.

    Notes:
        - The function assumes 'df' has a column named 'date'.
        - Non-trading days in 'df' are shifted forward to the next trading day.
        - The function relies on Yahoo Finance for trading day data and requires internet access.
        - The function will not work correctly if the Yahoo Finance API changes or becomes unavailable.
    """

    # Download stock data from Yahoo Finance
    print(f"Downloading trading days for {ticker} from {start_day} to {end_day}...")
    df_yf = yf.download(ticker, start=start_day, end=end_day, progress=False)
    df_yf = df_yf.reset_index()
    df_yf['Date'] = pd.to_datetime(df_yf['Date']).dt.normalize()

    # Convert the 'date' column of the input dataframe to datetime and normalize (remove time)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # Get the trading dates from the Yahoo Finance data as a set for faster lookup
    yf_date_set = set(df_yf['Date'])
    max_trading_date = df_yf['Date'].max()

    # Initialize an empty list to hold the adjusted dates
    trading = []

    # Loop over each date in the input dataframe
    for day in df['date']:
        original_day = day
        max_iterations = 365  # Prevent infinite loops
        iterations = 0

        # While the day is not a trading day, add one day
        while day not in yf_date_set:
            day += timedelta(days=1)
            iterations += 1

            # Safety check: if we've gone too far or past the max trading date
            if iterations > max_iterations or day > max_trading_date + timedelta(days=30):
                print(f"Warning: Could not find trading day for {original_day}, using closest: {max_trading_date}")
                day = max_trading_date
                break

        # Append the adjusted trading day to the list
        trading.append(day)

    # Assign the adjusted trading days back to the dataframe
    df['date'] = trading
    return df

def main(df, ticker, save_path, start_day, end_day):
    """
    Processes a DataFrame containing news data, cleans it, and adjusts its dates to match trading days.

    This function performs several operations on a news dataset:
    - Extracts update numbers and creates new headlines.
    - Removes duplicates and cleans the text.
    - Adjusts the dates to the nearest trading days based on stock data from Yahoo Finance.
    - Saves the cleaned and adjusted DataFrame to a CSV file.

    Input:
        df (pandas.DataFrame): The DataFrame containing news data.
        ticker (str): The stock ticker symbol for matching trading days.
        save_path (str): The file path where the cleaned DataFrame will be saved.
        start_day (str): The start date for the trading day range in 'YYYY-MM-DD' format.
        end_day (str): The end date for the trading day range in 'YYYY-MM-DD' format.

    Output:
        None: The function saves the cleaned DataFrame to a CSV file and prints summary information.
    """

    # Initial data processing steps: extract update numbers, create new headlines, and sort
    # Rename 'title' to 'headline' if needed for compatibility with functions
    if 'title' in df.columns and 'headline' not in df.columns:
        df = df.rename(columns={'title': 'headline'})

    df['update_number'] = df['headline'].apply(extract_update_number)
    df['new_headline'] = df.apply(create_new_headline, axis=1)
    df_sorted = df.sort_values(by='update_number', ascending=False)
    
    # Dropping duplicates and further cleaning
    df_drop_update = df_sorted.drop_duplicates(subset=['new_headline'], keep='first')
    df_drop_update = df_drop_update.sort_values(by='dates', ascending=False)
    df_drop_reuters = clean_news(df_drop_update)
    df_remove_spaces = remove_spaces(df_drop_reuters, ['new_headline', 'body'])

    # More deduplication and cleaning
    df_cleaned = df_remove_spaces.drop_duplicates(subset=['temp_body'], keep='first')
    df_cleaned = df_cleaned.drop_duplicates(subset=['temp_new_headline'], keep='first')
    df_final = df_cleaned.sort_values(by='dates')

    # Resetting symbols and timestamps, and cleaning text
    replace_column_values(df_final, 'symbols', ticker)
    df_final['dates'] = df_final['dates']
    df_final['dates'] = pd.to_datetime(df_final['dates'])
    df_final['date'] = df_final['dates'].apply(calculate_date)
    df_final['cleaned_body'] = df_final['body'].apply(clean_text)

    # Dropping duplicates post-cleaning
    df_final = df_final.drop_duplicates(subset=['cleaned_body'], keep='first')

    # Dropping temporary columns
    df_final = df_final.drop(columns=['item_id', 'update_number', 'headline', 'temp_body', 'temp_new_headline', 'body', 'dates'])

    # Save and read CSV for dropping similar records
    # If you employ the drop_similar_records function as shown below:
    # df_drop_similar = drop_similar_records(df_final, 'cleaned_body')
    # You will encounter a KeyError: 2.
    df_final.to_csv(save_path, index=False)
    df_similar = pd.read_csv(save_path)
    df_drop_similar = drop_similar_records(df_similar, 'cleaned_body', 0.6)
    df_drop_similar.to_csv(save_path, index=False)
    df_similar = pd.read_csv(save_path)
    df_drop_similar = drop_similar_records(df_similar, 'new_headline', 0.9)

    # Rearranging and renaming columns
    df_drop_similar = df_drop_similar[['date', 'symbols', 'new_headline', 'cleaned_body']]
    df_drop_similar = df_drop_similar.rename(columns={'new_headline': 'headline', 'cleaned_body': 'body'})

    # Matching to trading days and final save
    adjusted_df = adjust_trading_days(start_day, end_day, ticker, df_drop_similar)
    adjusted_df.to_csv(save_path, index=False)

    # Printing summary information
    print(f'{ticker} before cleaned:', len(df))
    print(f'{ticker} after cleaned:', len(df_drop_similar))

    # Uncomment to display the top 10 rows of the final DataFrame
    # print(df_drop_similar.head(10))

if __name__ == "__main__":
    # Read as Polars for metadata extraction
    df_pl = pl.read_parquet(os.path.join(RAW_DIR, "news.parquet"))
    start_date = df_pl["date"].min()
    end_date = df_pl["date"].max()

    # Convert to Pandas for processing (main function expects Pandas DataFrame)
    df_pd = df_pl.to_pandas()

    # Rename columns to match expected schema
    # Alpaca API uses 'datetime' but the cleaning script expects 'dates'
    if 'datetime' in df_pd.columns and 'dates' not in df_pd.columns:
        df_pd = df_pd.rename(columns={'datetime': 'dates'})

    # Rename 'equity' to 'symbols' if needed
    if 'equity' in df_pd.columns and 'symbols' not in df_pd.columns:
        df_pd = df_pd.rename(columns={'equity': 'symbols'})

    # Rename 'content' to 'body' if needed
    if 'summary' in df_pd.columns and 'body' not in df_pd.columns:
        df_pd = df_pd.rename(columns={'summary': 'body'})

    # Add item_id if not present (required for dropping columns later)
    if 'item_id' not in df_pd.columns:
        df_pd['item_id'] = range(len(df_pd))

    for ticker in df_pl.select("equity").unique().to_series().to_list():
        ticker_df = df_pd[df_pd['symbols'] == ticker].copy()
        save_path = os.path.join(RAW_DIR, f'cleaned_{ticker}_{start_date.strftime("%Y-%m-%d")}-{end_date.strftime("%Y-%m-%d")}.csv')
        main(ticker_df, ticker, save_path, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

# PFE,JPM,XOM,GS,C,MRNA,CVX,GM,F,MS,BAC,JNJ,WMT,NVDA,DIS,MRK

# TSLA: 6209
# AAPL: 4274
# AMZN: 4038
# PFE: 3762
# GOOG: 3600
# JPM: 3094
# XOM: 2895
# MSFT: 2883
# GS: 2786
# C: 2716
# MRNA: 2310
# CVX: 2110
# GM: 1927
# F: 1828
# BABA: 1717
# MS: 1666
# NFLX: 1655
# BAC: 1644
# JNJ: 1586
# WMT: 1540
# NVDA: 1295
# DIS: 1219
# MRK: 1131
# COIN: 896
# NIO: 484 
