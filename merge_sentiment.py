import pandas as pd

# Load sentiment scores
sentiment = pd.read_csv("Data/sentiment_scores.csv")
sentiment["headline"] = sentiment["headline"].astype(str).str.strip().str.lower()

# List of stock tickers and files
tickers = ["AAPL", "MSFT", "TSLA", "AMZN"]

for ticker in tickers:
    stock_path = f"Data/{ticker}_merged.csv"
    output_path = f"Data/{ticker}_final.csv"

    stock_df = pd.read_csv(stock_path)
    stock_df["headline"] = stock_df["headline"].astype(str).str.strip().str.lower()

    # Merge on 'headline'
    merged_df = pd.merge(stock_df, sentiment, on="headline", how="left")

    # Save merged file
    merged_df.to_csv(output_path, index=False)
    print(f"Merged and saved: {output_path}")
