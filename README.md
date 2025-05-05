# Hackathon May 2025 : Traffic Prediction 

<h2>🚗 Predicting Traffic Volume Using Weather and Time Data</h2>

Welcome! This project is all about predicting how busy a road will be (specifically, a highway called I-94 in the US) based on weather conditions, date, time, and whether it's a holiday.

We built two machine learning models — starting with a simple one and then improving it — to make these predictions.

<h2>🔍 What We’re Trying to Do</h2>

Every hour, the city records how many cars pass on a certain stretch of road. We want to build a model that can predict that traffic number, given things like:

- The temperature

- If it's raining or snowing

- How cloudy it is

- What day and time it is

- Whether it’s a holiday

<h2>📁 The Files You’ll See </h2>

- train_set_dirty.csv → The training data (includes traffic numbers we can learn from)

- test_set_nogt.csv → The test data (same format, but no traffic numbers)

- baseline_submission.csv → Predictions from the simple model

- rf_submission.csv → Predictions from the improved model

<h2>🛠️ How We Did It (Step by Step)</h2>

<h3>1. Loading the Data</h3>

- We loaded both the training and test files using Python’s pandas library.

<h3>2. Cleaning Things Up</h3>

- Some rows had missing values. We filled in missing numbers with the average and filled missing categories with the most common value.

<h3>3. Creating New Features (Feature Engineering)</h3>

From the date_time column, we extracted:

- Hour of the day (e.g. 8 AM or 5 PM)

- Day of the week (e.g. Monday, Saturday)

- Month (e.g. October)

- Whether it’s a weekend

These help the model understand traffic patterns — for example, traffic is usually heavier on weekdays around 9 AM and 5 PM.

<h3>4. Making Data Model-Ready</h3>

- We converted text columns (like “holiday” or “weather”) into numbers using something called one-hot encoding so the models can understand them.

<h2>🤖 The Models We Used</h2>

<h3> 1. Baseline Model: Linear Regression</h3>

This is a very simple model that tries to draw a straight line between inputs and the traffic volume. It's fast and easy to understand — great for starting out, but it can't handle complex patterns very well.

<h3>2. Improved Model: Random Forest</h3>

This is a much smarter model. Instead of one line, it builds lots of decision trees and combines them. It's better at handling:

- Nonlinear patterns (e.g., sudden increases in traffic during rain)

- Interactions between variables (e.g., rain + holiday might lower traffic more than either alone)

<h2>📊 How We Measured Success</h2>

We used two standard metrics:

- RMSE (Root Mean Squared Error) :	Tells how far off our predictions are — big mistakes hurt more.
- MAE (Mean Absolute Error) :	Average of how much we’re off by — easy to understand.

<h2>🏁 Results</h2>

- Linear Regression Model
  
       RMSE is 1823.55	
       MAE is 1591.31

- Random Forest Model

       RMSE is 1352.09
       MAE is 1102.80

<h2>✅ Improvement: </h2>

Random Forest did a much better job! It reduced the error by over 25%, showing that learning complex patterns in the data helps a lot.
