# Quick test to verify your setup before running the main notebook

import os
import pandas as pd

print("🔍 TESTING BASELINE MODELS SETUP")
print("="*50)

# 1. Check if required files exist
required_files = [
    '../Data/AAPL_merged.csv',
    '../Data/TSLA_merged.csv', 
    '../Data/MSFT_merged.csv',
    '../Data/AMZN_merged.csv'
]

print("\n📁 Checking required data files:")
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - NOT FOUND")

# 2. Check if sentiment scores exist
sentiment_files = [
    '../Data/sentiment_scores.csv',
    'sentiment_scores.csv',  # Alternative location
]

print("\n📊 Checking sentiment files:")
sentiment_found = False
for file in sentiment_files:
    if os.path.exists(file):
        print(f"✅ {file}")
        sentiment_found = True
        break

if not sentiment_found:
    print("⚠️  No sentiment_scores.csv found - will use label_num as fallback")

# 3. Test loading one dataset
print("\n📋 Testing data loading:")
try:
    df = pd.read_csv('../Data/AAPL_merged.csv')
    print(f"✅ Successfully loaded AAPL data: {len(df):,} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['date', 'open', 'close', 'movement', 'headline', 'label', 'label_num']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")
        
except Exception as e:
    print(f"❌ Error loading AAPL data: {e}")

# 4. Check required packages
print("\n📦 Checking required packages:")
required_packages = [
    'pandas', 'numpy', 'matplotlib', 'seaborn', 
    'sklearn', 'warnings'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} - INSTALL REQUIRED")

print("\n🚀 Setup check complete!")
print("\nIf all files are found and packages installed, you're ready to run the baseline models notebook!")