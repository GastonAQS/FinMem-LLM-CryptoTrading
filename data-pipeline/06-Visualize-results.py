import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def get_data(Start: str, End: str, ticker: str, df_dict: dict, col: list):
    """
    Downloads stock data from Yahoo Finance and filters dataframes based on the given date range.

    Parameters:
        Start (str): Start date for the data range in 'YYYY-MM-DD' format.
        End (str): End date for the data range in 'YYYY-MM-DD' format.
        ticker (str): Stock ticker symbol to download data for.
        df_dict (dict): Dictionary of dataframes to be filtered.
        col (list): List containing the names of the columns to filter by.

    Returns:
        tuple: A tuple containing the adjusted closing prices of the stock, 
               list of model names, and a list of filtered data corresponding to each model.
    """
    # Download stock data from Yahoo Finance
    df = yf.download(ticker, start=Start, end=End)

    # Dictionary to hold filtered data
    model_name = []
    filtered_data = []

    for df_name, dataframe in df_dict.items():
        model_name.append(df_name)
        dataframe[col[0]] = pd.to_datetime(dataframe[col[0]])
        filtered_data.append(dataframe[(dataframe[col[0]] >= Start) & (dataframe[col[0]] < End)][col[1]].tolist())

    return df['Adj Close'].tolist(), model_name, filtered_data
    

def reward_list(price: list, actions: list):
    """
    Calculates the cumulative reward for a given list of prices and actions.

    Parameters:
        price (list): List of stock prices.
        actions (list): List of actions taken on the stock.

    Returns:
        list: List of cumulative rewards calculated from the prices and actions.
    """
    reward = 0
    reward_list = [0]
    for i in range(len(price)-1):
        reward += actions[i] * np.log(price[i+1]/price[i])
        reward_list.append(reward)
    return reward_list
    

def plot_cumulative_returns(dates, return_lists, labels, colors, linestyles, alphas, linewidths, ticker, file_path, Start_Date=True):
    """
    Plots cumulative returns using the provided data.

    Parameters:
        dates (list): List of dates for the x-axis.
        return_lists (list of lists): List of return lists to be plotted.
        labels (list): List of labels for each return series.
        colors (list): List of colors for each return series.
        linestyles (list): List of line styles for each return series.
        alphas (list): List of alpha values for each return series.
        linewidths (list): List of line widths for each return series.
        ticker (str): Stock ticker symbol for the title.
        file_path (str): Path to save the plot.
        Start_Date (bool, optional): Whether to start x-ticks from a specific date. Defaults to True.

    Returns:
        None: The function generates and displays a plot.
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Loop through the return lists and plot each one
    for returns, label, color, linestyle, alpha, linewidth in zip(return_lists, labels, colors, linestyles, alphas, linewidths):
        ax.plot(dates, returns, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
    
    # Set the labels and title
    ax.set_xlabel('Date', fontsize=28)
    ax.set_ylabel('Cumulative Return', fontsize=28)
    plt.title(ticker, fontsize=35)

    # Customize the legend
    ax.legend(fontsize=22, frameon=True)

    # Customize the grid
    ax.grid(True)

    # Customize the tick labels on both axes
    ax.tick_params(axis='x', labelsize=22, width=2, rotation=45)  # Rotate x-axis labels
    ax.tick_params(axis='y', labelsize=22, width=2)  # y-axis labels

    
    # Set x-ticks to start from a specific date
    if Start_Date:
        start_date = datetime(2022, 10, 1)
        ax.set_xlim(left=start_date)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to monthly intervals
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())  # Set minor ticks to weekly intervals

    # Apply tight layout to ensure everything fits without overlapping
    plt.tight_layout()

    # Save the plot as a .png file with 300 dpi
    # plt.savefig(file_path, format='png', dpi=300)

    # Display the plot
    plt.show()

def main(Ticker, start_time, end_time, df_dict, col, image_save_path, Start_Date=True):
    """
    Main function to process data and plot cumulative returns.

    Parameters:
        Ticker (str): Stock ticker symbol.
        start_time (str): Start date for data processing in 'YYYY-MM-DD' format.
        end_time (str): End date for data processing in 'YYYY-MM-DD' format.
        df_dict (dict): Dictionary of dataframes to process.
        col (list): List of column names to use in processing.
        image_save_path (str): Path to save the generated plot.
        Start_Date (bool, optional): Whether to start x-ticks from a specific date. Defaults to True.

    Returns:
        None: This function processes data and generates a plot.
    """
    # Get data
    price, labels, actions_list = get_data(
        Start=start_time, 
        End=end_time, 
        ticker=Ticker, 
        df_dict=df_dict,
        col=col
    )
    labels = ['B_H'] + labels
    return_lists = []
    # Calculate rewards
    B_H = [1] * len(price)
    B_H_rw = reward_list(price, B_H)
    return_lists.append(B_H_rw)
    for actions in actions_list:
        rw = reward_list(price, actions)
        return_lists.append(rw)
    # Prepare data for plotting
    data = yf.download(Ticker, start=start_time, end=end_time)
    data = data.reset_index()
    # data['Date'] = data['Date'].dt.date
    Date = data[(data['Date'] >= start_time) & (data['Date'] < end_time)]['Date'].tolist()
    dates = pd.to_datetime(Date).tolist()
    # print(len(Date), len(B_H_rw))
    colors = ['#000', '#d14749', '#59a14f', '#4e89e0', '#ee4199', '#f28e2b', '#8F337F'] # match the lenth of len(df_dict)+1
    linestyles = ['-.', '-', '-', '-', '-', '-', '-'] # match the lenth of len(df_dict)+1
    linewidths = [2.5, 3.5, 2.5, 2.5, 2.5, 2.5, 2.5] # match the lenth of len(df_dict)+1
    alphas = [0.66, 1, 1, 1, 0.66, 0.66, 0.66] # match the lenth of len(df_dict)+1

    # Plot the cumulative returns
    plot_cumulative_returns(
        dates=dates,
        return_lists=return_lists,
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        alphas=alphas,
        linewidths=linewidths,
        ticker=Ticker,
        file_path=image_save_path,
        Start_Date = Start_Date
    )

if __name__ == '__main__':
    from pathlib import Path

    # Get project root directory
    PROJECT_ROOT = Path(__file__).parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    # Create directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dictionary containing file paths
    """
    file_path = {Model_name: path of model output actions}

    IMPORTANT: Each CSV file must have these columns:
    - 'date': Trading date (YYYY-MM-DD format)
    - 'direction': Trading action (typically 1 for buy/hold, 0 for sell, or -1/0/1 for short/hold/long)

    You need to run your trading model/agent first to generate these CSV files.
    The CSV files contain the trading decisions made by each model on each date.
    """

    # Update these paths to your actual model output files
    file_paths = {
        # 'FinMem': str(RESULTS_DIR / 'FinMem_TSLA.csv'),
        # 'YourModel': str(RESULTS_DIR / 'YourModel_TSLA.csv'),
        # Add your model result files here
    }

    # Example: If you have result files, uncomment and update:
    # file_paths = {
    #     'Model1': str(RESULTS_DIR / 'model1_actions.csv'),
    #     'Model2': str(RESULTS_DIR / 'model2_actions.csv'),
    # }

    if not file_paths:
        print("=" * 60)
        print("NO MODEL RESULTS FOUND")
        print("=" * 60)
        print("\nThis script visualizes trading strategy performance.")
        print("You need to:")
        print("1. Run your trading model/agent to generate action CSV files")
        print("2. Each CSV must have columns: ['date', 'direction']")
        print("3. Update file_paths dictionary with your result file paths")
        print("\nExample CSV format:")
        print("  date,direction")
        print("  2024-11-15,1")
        print("  2024-11-18,1")
        print("  2024-11-19,0")
        print("\nWhere direction is the trading action:")
        print("  1 = Buy/Long position")
        print("  0 = Hold/No position")
        print("  -1 = Sell/Short position (optional)")
        print("=" * 60)
        exit(0)

    # Loading DataFrames from the file paths
    df_dict = {key: pd.read_csv(path) for key, path in file_paths.items()}

    # Configuration - Update these to match your data
    Ticker = 'TSLA'
    start_time = '2024-11-15'  # Match your data range
    end_time = '2024-12-30'
    col = ['date', 'direction']  # Column names in your CSV files
    image_save_path = str(OUTPUT_DIR / f'{Ticker}_{start_time}_{end_time}.png')
    Start_Date = False

    # Main function call
    main(Ticker, start_time, end_time, df_dict, col, image_save_path, Start_Date)
