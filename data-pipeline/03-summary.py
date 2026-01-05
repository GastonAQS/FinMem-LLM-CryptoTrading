from model_wrapper import ModelFactory
import pandas as pd
import concurrent.futures
import threading
from pathlib import Path




model = ModelFactory.create_model('llama')
# Global lock to ensure thread-safe access to the llama model
model_lock = threading.Lock()

def process_row(row, lock, df, df_name):
    # Perform summary on the 'body' column with thread-safe model access
    with model_lock:
        result = model.summarize(row['body'])

    # Acquire lock before updating the dataframe and saving to CSV
    with lock:
        # Update the 'summary' column
        df.at[row.name, 'summary'] = result

        # Save the dataframe to CSV
        df.to_csv(TEMP_PATH / df_name, index=False)
        print("saving")

# Function to parallelize the summary tasks using threads
def parallel_summary(df, df_name):
    lock = threading.Lock()
    df_copy = df.copy()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each row's summary task to the thread pool
        futures = [executor.submit(process_row, row, lock, df_copy, df_name) for _, row in df.iterrows()]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    return df_copy


# Get project root directory (parent of data-pipeline)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_PATH = DATA_DIR / "01_raw"
DEST_PATH = DATA_DIR / "02_processed"
TEMP_PATH = DATA_DIR / "temp"

# Create directories if they don't exist
DEST_PATH.mkdir(parents=True, exist_ok=True)
TEMP_PATH.mkdir(parents=True, exist_ok=True)


def process_main(file):
    if file.endswith('.csv'):
        print(f"Processing: {file}")
        df = pd.read_csv(SOURCE_PATH / file)
        df["summary"] = None
        ret = parallel_summary(df, file)
        ret.to_csv(DEST_PATH / file)
        print(f"New DF dumped to {DEST_PATH / file}")


if __name__ == "__main__":
    file_ls = list(SOURCE_PATH.iterdir())

    for file_path in file_ls:
        if file_path.suffix == '.csv':
            process_main(file_path.name)
