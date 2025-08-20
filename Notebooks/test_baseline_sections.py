# Test the baseline models notebook in sections

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

print("🧪 TESTING BASELINE MODELS - SECTION BY SECTION")
print("="*60)

# TEST SECTION 1: Data Loading
print("\n1️⃣ TESTING DATA LOADING")
print("-" * 30)

try:
    # Load one stock dataset
    df = pd.read_csv('../Data/AAPL_merged.csv')
    print(f"✅ Loaded AAPL: {len(df):,} rows")
    
    # Check data quality
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Sample movement: {df['movement'].describe()}")
    
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    exit()

# TEST SECTION 2: Feature Creation
print("\n2️⃣ TESTING FEATURE CREATION")
print("-" * 30)

try:
    # Create basic features
    df['movement_pct'] = (df['movement'] / df['open']) * 100
    df['price_direction'] = (df['movement'] > 0).astype(int)
    
    # Use label_num as sentiment (fallback if no advanced sentiment)
    df['sentiment_score'] = df['label_num']
    
    print(f"✅ Created features")
    print(f"   Price direction distribution: {df['price_direction'].value_counts().to_dict()}")
    print(f"   Movement % range: {df['movement_pct'].min():.2f}% to {df['movement_pct'].max():.2f}%")
    
except Exception as e:
    print(f"❌ Feature creation failed: {e}")
    exit()

# TEST SECTION 3: Model Training
print("\n3️⃣ TESTING MODEL TRAINING")
print("-" * 30)

try:
    # Prepare simple features
    X = df[['sentiment_score', 'label_num']].fillna(0)
    y_class = df['price_direction']
    y_reg = df['movement_pct']
    
    # Test train/test split
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    
    print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Test classification
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train_class)
    y_pred_class = clf.predict(X_test)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    
    print(f"✅ Classification test: {accuracy:.3f} accuracy")
    
    # Test regression
    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    reg = LinearRegression()
    reg.fit(X_train, y_train_reg)
    y_pred_reg = reg.predict(X_test)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"✅ Regression test: {r2:.3f} R²")
    
except Exception as e:
    print(f"❌ Model training failed: {e}")
    print(f"   Error details: {type(e).__name__}")

# TEST SECTION 4: Quick Results
print("\n4️⃣ QUICK RESULTS SUMMARY")
print("-" * 30)

print(f"📊 Dataset size: {len(df):,} records")
print(f"📈 Price up: {(df['price_direction'] == 1).sum():,} ({(df['price_direction'] == 1).mean():.1%})")
print(f"📉 Price down: {(df['price_direction'] == 0).sum():,} ({(df['price_direction'] == 0).mean():.1%})")
print(f"🎯 Classification accuracy: {accuracy:.3f}")
print(f"📏 Regression R²: {r2:.3f}")

print(f"\n✅ SECTION TESTING COMPLETE!")
print("If all sections passed, your full notebook should work!")

# Expected results guidance
print(f"\n💡 EXPECTED RESULTS GUIDE:")
print("✅ Good signs:")
print("   - Classification accuracy > 0.50 (better than random)")
print("   - Regression R² > 0.01 (some predictive power)")
print("   - No major errors in data loading/processing")
print("\n⚠️  If results are poor:")
print("   - This is normal for baseline models!")
print("   - Week 5 advanced models should improve performance")
print("   - Focus on whether the code runs without errors")