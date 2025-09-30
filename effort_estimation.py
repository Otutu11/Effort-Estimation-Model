import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic project data for effort estimation.
    Returns a DataFrame with project features and actual effort.
    """
    # Project size (in function points or story points)
    size = np.random.lognormal(mean=3, sigma=0.5, size=n_samples).round(1)
    
    # Team experience (years)
    team_exp = np.random.normal(loc=5, scale=2, size=n_samples).round(1)
    team_exp = np.clip(team_exp, 1, 10)  # clip between 1-10 years
    
    # Requirements volatility (scale 1-5)
    req_volatility = np.random.randint(1, 6, size=n_samples)
    
    # Technical complexity (scale 1-5)
    tech_complexity = np.random.randint(1, 6, size=n_samples)
    
    # Number of team members
    team_size = np.random.randint(2, 11, size=n_samples)
    
    # Use of methodology (1=waterfall, 2=agile, 3=hybrid)
    methodology = np.random.choice([1, 2, 3], size=n_samples, p=[0.3, 0.5, 0.2])
    
    # Generate actual effort based on features with some noise
    base_effort = (size * 0.5 + 
                   team_size * 2 + 
                   tech_complexity * 5 - 
                   team_exp * 0.8 + 
                   req_volatility * 3)
    
    # Add methodology factor
    methodology_factor = np.where(methodology == 1, 1.1, 
                                 np.where(methodology == 2, 0.9, 1.0))
    base_effort = base_effort * methodology_factor
    
    # Add some noise
    noise = np.random.normal(0, 10, size=n_samples)
    actual_effort = np.clip(base_effort + noise, 20, 200).round(1)
    
    # Create DataFrame
    data = pd.DataFrame({
        'project_size': size,
        'team_experience': team_exp,
        'requirements_volatility': req_volatility,
        'technical_complexity': tech_complexity,
        'team_size': team_size,
        'methodology': methodology,
        'actual_effort': actual_effort
    })
    
    return data

# Generate the synthetic dataset
project_data = generate_synthetic_data(1500)

# Explore the data
print(project_data.head())
print("\nData Description:")
print(project_data.describe())

# Visualize relationships
plt.figure(figsize=(12, 8))
sns.pairplot(project_data, 
             vars=['project_size', 'team_experience', 'technical_complexity', 'actual_effort'],
             hue='methodology',
             plot_kws={'alpha': 0.6})
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

# Preprocess data
# Convert methodology to dummy variables
data_processed = pd.get_dummies(project_data, columns=['methodology'], drop_first=True)

# Split data into features and target
X = data_processed.drop('actual_effort', axis=1)
y = data_processed['actual_effort']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f} person-days")
print(f"R-squared: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Effort Estimation')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Example prediction
sample_project = pd.DataFrame({
    'project_size': [75],
    'team_experience': [4.5],
    'requirements_volatility': [3],
    'technical_complexity': [4],
    'team_size': [5],
    'methodology_2': [1],  # agile
    'methodology_3': [0]   # not hybrid
})

predicted_effort = model.predict(sample_project)
print(f"\nExample Project Effort Prediction: {predicted_effort[0]:.1f} person-days")


# Save model (uncomment to use)
# import joblib
# joblib.dump(model, 'effort_estimation_model.pkl')
