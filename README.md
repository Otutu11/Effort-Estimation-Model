# Effort Estimation Model

A machine learning model for predicting project effort based on various project characteristics, using synthetic data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Generation](#data-generation)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Example](#example)
- [License](#license)

## Overview

This project implements a Random Forest regression model to estimate project effort (in person-days) based on synthetic project data. The model takes into account factors like project size, team experience, technical complexity, and development methodology.

## Features

- Synthetic data generation with customizable parameters
- Exploratory data analysis visualizations
- Random Forest regression model
- Feature importance analysis
- Model evaluation metrics
- Example prediction capability

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Akajiaku1/effort-estimation-model.git
   cd effort-estimation-model

    Create and activate a virtual environment (recommended):
    bash
    Copy

    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    Install the required packages:
    bash
    Copy

    pip install -r requirements.txt

    Or install them manually:
    bash
    Copy

    pip install numpy pandas scikit-learn matplotlib seaborn

Usage

Run the main script:
bash
Copy

python effort_estimation.py

This will:

    Generate synthetic project data

    Train the effort estimation model

    Evaluate the model performance

    Show feature importance

    Make an example prediction

Data Generation

The synthetic data includes these features:
Feature	Description	Range/Values
project_size	Project size in function/story points	Log-normal distribution
team_experience	Average team experience in years	1-10 years
requirements_volatility	Requirements stability	1-5 scale
technical_complexity	Technical difficulty	1-5 scale
team_size	Number of team members	2-10 people
methodology	Development methodology	1=Waterfall, 2=Agile, 3=Hybrid
actual_effort	Actual effort in person-days	20-200 days
Model Details

    Algorithm: Random Forest Regressor

    Hyperparameters:

        n_estimators: 100

        random_state: 42

    Input Features: All features except actual_effort

    Target Variable: actual_effort

Evaluation Metrics

The model is evaluated using:

    Mean Absolute Error (MAE): Average absolute difference between predictions and actual values

    R-squared (R²): Proportion of variance in the dependent variable that's predictable

Typical performance on synthetic data:

    MAE: ~8-12 person-days

    R²: ~0.85-0.95
