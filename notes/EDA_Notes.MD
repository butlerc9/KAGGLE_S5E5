# Feature Engineering
- Convert the sex column to a binary flag for is_male.

# EDA notes
- There is actually no missing data, so literally no cleaning required
- Calories is right skewed with a fast drop off
- Calories is apparently highly correlated to body_temp, duration and heart rate

# Overall plan
- Tune a model with Optuna
- Have a series of XGBoost model parameters