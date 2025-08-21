#!/usr/bin/env python3
"""
Simple Baseline Models - Week 4
Simplified version with auto-path detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

print("🚀 SIMPLE BASELINE MODELS ANALYSIS")
print("="*50)

# Step 1: Load one dataset to test
print("\n1. Loading test data...")

# Try to find AAPL file
possible_paths = [
    '../Data/AAPL_merged.csv',
    'Data/AAPL_merged.csv', 
    'AAPL_merged.csv',
    '../AAPL_merged.csv'
]

df = None
used_path = None

for path in possible_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            used_path = path
            print(f"✅ Found data at: {path}")
            break
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
            continue

if df is None:
    print("❌ Could not find AAPL_merged.csv in any location")
    print("Checked these paths:")
    for path in possible_paths:
        print(f"   - {path} ({'EXISTS' if os.path.exists(path) else 'NOT FOUND'})")
    exit()

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Step 2: Create simple features
print("\n2. Creating features...")

# Basic features
df['movement_pct'] = (df['movement'] / df['open']) * 100
df['price_direction'] = (df['movement'] > 0).astype(int)

# Simple features for modeling
X = pd.DataFrame({
    'sentiment': df['label_num'],  # -1, 0, 1
    'sentiment_abs': abs(df['label_num']),  # 0, 1, 1
    'day_of_week': pd.to_datetime(df['date']).dt.dayofweek
})

y_class = df['price_direction']  # Binary target
y_reg = df['movement_pct']       # Continuous target

print(f"✅ Features created:")
print(f"   X shape: {X.shape}")
print(f"   Price up: {(y_class == 1).sum()} ({(y_class == 1).mean():.1%})")
print(f"   Price down: {(y_class == 0).sum()} ({(y_class == 0).mean():.1%})")

# Step 3: Train classification model
print("\n3. Training classification model...")

# Split data
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_scaled, y_train_class)

# Predictions
y_pred_class = log_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test_class, y_pred_class)

print(f"✅ Logistic Regression Results:")
print(f"   Accuracy: {accuracy:.3f}")
print(f"   Better than random: {'✅' if accuracy > 0.5 else '❌'}")

# Step 4: Train regression model
print("\n4. Training regression model...")

# Split data for regression
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear regression
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train_reg)

# Predictions
y_pred_reg = reg_model.predict(X_test_scaled)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"✅ Linear Regression Results:")
print(f"   R² Score: {r2:.3f}")
print(f"   Has predictive power: {'✅' if r2 > 0.001 else '❌'}")

# Step 5: Summary
print(f"\n5. WEEK 4 BASELINE SUMMARY")
print("="*50)
print(f"📊 Dataset: {len(df):,} records (AAPL stock)")
print(f"📁 Data source: {used_path}")
print(f"🎯 Classification Accuracy: {accuracy:.3f}")
print(f"📏 Regression R²: {r2:.3f}")

print(f"\n💡 BASELINE PERFORMANCE:")
if accuracy > 0.55:
    print("✅ Classification: Good predictive power")
elif accuracy > 0.5:
    print("⚠️  Classification: Modest predictive power")
else:
    print("❌ Classification: Poor performance")

if r2 > 0.01:
    print("✅ Regression: Some predictive power")
else:
    print("❌ Regression: Limited predictive power")

print(f"\n🎉 WEEK 4 BASELINE MODELS COMPLETE!")
print("✅ Task 1: Trained baseline models (Logistic + Linear Regression)")
print("✅ Task 2: Generated metrics and documented results")
print("\nThese results establish your baseline for Week 5 advanced modeling!")
