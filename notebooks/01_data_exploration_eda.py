"""
Module 1: Data Exploration and EDA
===================================
This notebook covers:
1. Loading the LendingClub dataset
2. Understanding the target variable (loan_status)
3. Exploratory Data Analysis
4. Identifying data quality issues

Author: Your Name
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CREATE PROJECT DIRECTORY STRUCTURE
# ============================================================================
# Get the current working directory
PROJECT_DIR = Path.cwd()
print(f"Working Directory: {PROJECT_DIR}")

# Create subdirectories if they don't exist
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"

for directory in [RESULTS_DIR, DATA_DIR, DOCS_DIR]:
    directory.mkdir(exist_ok=True)
    
print(f"✓ Created project structure:")
print(f"  - Results: {RESULTS_DIR}")
print(f"  - Data: {DATA_DIR}")
print(f"  - Docs: {DOCS_DIR}")

print("\n" + "="*60)
print("LOAN DEFAULT RISK ASSESSMENT - MODULE 1")
print("Data Exploration & EDA")
print("="*60)

# ============================================================================
# STEP 1: LOAD THE DATASET
# ============================================================================

print("\n[STEP 1] Loading the LendingClub Dataset...")
print("-" * 60)

# NOTE: Update this path to where your dataset is located
# You should have 'accepted_2007_to_2018Q4.csv' downloaded from Kaggle
DATA_PATH = r"E:\TMU\MRP\dataset\LOAN_DATA\accepted_2007_to_2018Q4.csv"

# Load only first 100k rows for initial exploration (full dataset is huge ~2GB)
# Remove nrows parameter to load full dataset later
df = pd.read_csv(DATA_PATH, nrows=100000, low_memory=False)

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# STEP 2: INITIAL DATA INSPECTION
# ============================================================================

print("\n[STEP 2] Initial Data Inspection")
print("-" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# STEP 3: UNDERSTANDING THE TARGET VARIABLE
# ============================================================================

print("\n[STEP 3] Understanding the Target Variable: loan_status")
print("-" * 60)

# The loan_status column tells us if a loan defaulted or not
print("\nUnique loan statuses:")
print(df['loan_status'].value_counts())

"""
EXPLANATION: Target Variable Definition
----------------------------------------
In loan default prediction, we need to define what counts as "default":

Typically:
- DEFAULT (1): Charged Off, Default, Does not meet credit policy (Charged Off)
- NON-DEFAULT (0): Fully Paid, Current, Does not meet credit policy (Fully Paid)

We'll exclude: In Grace Period, Late payments (for now - can be separate analysis)
"""

# Create binary target variable
def create_target(status):
    """
    Convert loan_status to binary classification
    1 = Default (bad loan)
    0 = Paid (good loan)
    """
    default_status = ['Charged Off', 'Default', 
                      'Does not meet the credit policy. Status:Charged Off']
    return 1 if status in default_status else 0

df['default'] = df['loan_status'].apply(create_target)

# Filter to keep only clear cases (exclude current, late, etc.)
clear_statuses = ['Fully Paid', 'Charged Off', 'Default',
                  'Does not meet the credit policy. Status:Fully Paid',
                  'Does not meet the credit policy. Status:Charged Off']
df_filtered = df[df['loan_status'].isin(clear_statuses)].copy()

print(f"\n✓ Filtered dataset: {df_filtered.shape[0]:,} loans with clear outcomes")
print("\nTarget Variable Distribution:")
print(df_filtered['default'].value_counts())
print("\nPercentages:")
print(df_filtered['default'].value_counts(normalize=True) * 100)

"""
IMPORTANT CONCEPT: Class Imbalance
-----------------------------------
Notice that defaults are typically 15-25% of loans.
This is IMBALANCED data - we'll need to handle this in Module 2!
"""

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS - KEY FEATURES
# ============================================================================

print("\n[STEP 4] Exploring Key Features")
print("-" * 60)

# Select important columns for loan risk assessment
important_features = [
    'loan_amnt', 'term', 'int_rate', 'installment', 
    'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'annual_inc', 'verification_status', 'purpose',
    'dti', 'delinq_2yrs', 'earliest_cr_line', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'default'
]

# Keep only features that exist in the dataset
available_features = [f for f in important_features if f in df_filtered.columns]
df_analysis = df_filtered[available_features].copy()

print(f"\n✓ Selected {len(available_features)} key features for analysis")

# ============================================================================
# STEP 5: MISSING DATA ANALYSIS
# ============================================================================

print("\n[STEP 5] Missing Data Analysis")
print("-" * 60)

missing_data = pd.DataFrame({
    'Column': df_analysis.columns,
    'Missing_Count': df_analysis.isnull().sum(),
    'Missing_Percentage': (df_analysis.isnull().sum() / len(df_analysis)) * 100
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
    'Missing_Percentage', ascending=False
)

print("\nFeatures with missing values:")
print(missing_data)

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n[STEP 6] Creating Visualizations...")
print("-" * 60)

# Visualization 1: Target Distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df_analysis['default'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Loan Status Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Default Status (0=Paid, 1=Default)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df_analysis['default'].value_counts(normalize=True).plot(kind='pie', 
                                                           autopct='%1.1f%%',
                                                           colors=['green', 'red'])
plt.title('Loan Status Percentage', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig(RESULTS_DIR / '01_target_distribution.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: 01_target_distribution.png")

# Visualization 2: Loan Grade Distribution
if 'grade' in df_analysis.columns:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df_analysis['grade'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Loan Grades', fontsize=14, fontweight='bold')
    plt.xlabel('Grade')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    grade_default = df_analysis.groupby('grade')['default'].mean().sort_index()
    grade_default.plot(kind='bar', color='coral')
    plt.title('Default Rate by Loan Grade', fontsize=14, fontweight='bold')
    plt.xlabel('Grade')
    plt.ylabel('Default Rate')
    plt.axhline(y=df_analysis['default'].mean(), color='red', linestyle='--', 
                label='Overall Default Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '02_grade_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_grade_analysis.png")

# Visualization 3: Numerical Features Distribution
numerical_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
available_numerical = [f for f in numerical_features if f in df_analysis.columns]

if available_numerical:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(available_numerical[:4]):
        # Box plot by default status
        df_analysis.boxplot(column=feature, by='default', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Default Status')
        axes[idx].set_xlabel('Default (0=No, 1=Yes)')
        
    plt.suptitle('Key Numerical Features by Default Status', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '03_numerical_features.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_numerical_features.png")

# ============================================================================
# STEP 7: CORRELATION ANALYSIS
# ============================================================================

print("\n[STEP 7] Correlation Analysis")
print("-" * 60)

# Select only numerical columns for correlation
numerical_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()

if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_analysis[numerical_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '04_correlation_matrix.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_correlation_matrix.png")
    
    # Show top correlations with target
    target_corr = correlation_matrix['default'].sort_values(ascending=False)
    print("\nTop 10 features correlated with Default:")
    print(target_corr.head(11))  # 11 because 'default' itself is included

# ============================================================================
# STEP 8: SUMMARY STATISTICS BY TARGET
# ============================================================================

print("\n[STEP 8] Summary Statistics by Default Status")
print("-" * 60)

# Compare statistics between defaulted and non-defaulted loans
print("\nNumerical Features - Default vs Non-Default:")
summary_by_target = df_analysis.groupby('default')[numerical_cols].mean()
print(summary_by_target)

# ============================================================================
# STEP 9: SAVE PROCESSED DATA
# ============================================================================

print("\n[STEP 9] Saving Processed Data")
print("-" * 60)

# Save the filtered dataset for next module
df_analysis.to_csv(DATA_DIR / 'loans_filtered.csv', 
                   index=False)
print(f"✓ Saved filtered dataset: {df_analysis.shape[0]:,} rows")

# Create a summary report
# Calculate missing data info
most_missing_col = missing_data.iloc[0]['Column'] if len(missing_data) > 0 else 'None'
most_missing_pct = missing_data.iloc[0]['Missing_Percentage'] if len(missing_data) > 0 else 0

default_rate = df_analysis['default'].mean() * 100
non_default_count = (df_analysis['default']==0).sum()
non_default_pct = non_default_count / len(df_analysis) * 100
default_count = (df_analysis['default']==1).sum()
default_pct = default_count / len(df_analysis) * 100

summary_report = f"""
LOAN DEFAULT RISK ASSESSMENT - MODULE 1 SUMMARY
================================================

Dataset Overview:
- Total Loans Analyzed: {df_analysis.shape[0]:,}
- Total Features: {df_analysis.shape[1]}
- Default Rate: {default_rate:.2f}%

Target Variable:
- Non-Default (0): {non_default_count:,} ({non_default_pct:.1f}%)
- Default (1): {default_count:,} ({default_pct:.1f}%)

Missing Data:
- Features with missing values: {len(missing_data)}
- Most missing: {most_missing_col} ({most_missing_pct:.1f}%)

Key Insights:
1. Class Imbalance: The dataset shows typical loan portfolio characteristics with more paid loans than defaults
2. Missing Data: Several features have missing values that need handling in Module 2
3. Feature Correlations: Some features show correlation with default status
4. Loan Grade: Clear relationship between grade and default rate

Next Steps (Module 2):
- Handle missing values
- Engineer new features
- Create cybersecurity-inspired behavioral features
- Address class imbalance
- Prepare data for modeling
"""

with open(DOCS_DIR / 'module1_summary.txt', 'w') as f:
    f.write(summary_report)

print("\n" + "="*60)
print("MODULE 1 COMPLETED!")
print("="*60)
print(f"\n✓ All visualizations saved to: {RESULTS_DIR}")
print(f"✓ Processed data saved to: {DATA_DIR / 'loans_filtered.csv'}")
print(f"✓ Summary report saved to: {DOCS_DIR / 'module1_summary.txt'}")
print("\n📚 KEY LEARNINGS:")
print("   1. Target variable definition is crucial")
print("   2. Class imbalance is a common challenge")
print("   3. Missing data patterns inform preprocessing strategy")
print("   4. EDA reveals feature-target relationships")
print("\n👉 When ready, let me know to proceed to MODULE 2!")
