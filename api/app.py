from flask import Flask, request, Response, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import datetime
from random import randint
import io
import uuid
import time
import logging
import traceback
import re
from collections import Counter
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)



# Store streamed messages with thread-safe access
stream_messages = []
stream_id = str(uuid.uuid4())  # Unique ID for each upload session

def generate_stream():
    """Generate stream of messages for the current session."""
    last_sent = -1
    while True:
        if len(stream_messages) > last_sent + 1:
            last_sent += 1
            yield f"data: {stream_messages[last_sent]}\n\n"
            time.sleep(0.1)
        elif len(stream_messages) > 0 and stream_messages[0].startswith("Error"):
            yield f"data: {stream_messages[0]}\n\n"
            break
        else:
            time.sleep(0.5)

@app.route('/')
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route('/previous_data.xlsx')
def download_sample():
    return send_from_directory('static', 'previous_data.xlsx')


@app.route('/stream')
def stream():
    """Stream analysis results."""
    return Response(generate_stream(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and perform analysis."""
    global stream_messages, stream_id
    stream_messages = ["CLEAR"]  # Clear previous messages
    stream_id = str(uuid.uuid4())  # New session ID
    logger.info("New file upload initiated")

    if 'file' not in request.files:
        error_msg = "Error: No file selected for upload"
        stream_messages.append(error_msg)
        logger.error(error_msg)
        return render_template("index.html")

    file = request.files['file']
    if not file.filename.endswith('.xlsx'):
        error_msg = "Error: Please upload an Excel (.xlsx) file"
        stream_messages.append(error_msg)
        logger.error(error_msg)
        return render_template("index.html")

    if not file:
        error_msg = "Error: Empty file uploaded"
        stream_messages.append(error_msg)
        logger.error(error_msg)
        return render_template("index.html")

    try:
        # Read the Excel file
        logger.info("Reading uploaded Excel file")
        data = pd.read_excel(file)
        stream_messages.append(f"File uploaded successfully. Rows: {len(data)}")
        stream_messages.append(f"Column names: {data.columns.tolist()}")
        logger.info(f"File loaded with {len(data)} rows and columns: {data.columns.tolist()}")

        # Identify potential date columns with improved detection
        date_cols = []
        for col in data.columns:
            # Skip columns that are completely empty
            if data[col].isna().all():
                continue
                
            # Convert to string for regex matching
            col_as_str = data[col].astype(str)
            
            # Check for various date patterns
            has_dates = (
                # YYYY-Month-DD pattern (with or without day name)
                col_as_str.str.contains(r'\d{4}-[A-Za-z]+-\d+', regex=True).any() or
                # Month DD, YYYY pattern (with or without day name)
                col_as_str.str.contains(r'[A-Za-z]+ \d+,? \d{4}', regex=True).any() or
                # DD Month YYYY pattern
                col_as_str.str.contains(r'\d+ [A-Za-z]+ \d{4}', regex=True).any() or
                # Standard YYYY-MM-DD pattern
                col_as_str.str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).any() or
                # MM/DD/YYYY or DD/MM/YYYY pattern
                col_as_str.str.contains(r'\d{1,2}/\d{1,2}/\d{4}', regex=True).any() or
                # Day Month DD, YYYY pattern
                col_as_str.str.contains(r'[A-Za-z]+ [A-Za-z]+ \d+,? \d{4}', regex=True).any()
            )
            
            if has_dates:
                date_cols.append(col)
        
        # Identify numeric and letter columns
        valid_numeric_cols = []
        letter_cols = []
        for col in data.columns:
            if col in date_cols:
                continue
            if col.startswith('Unnamed') and data[col].isna().all():
                continue
                
            # Check for column names that suggest bonus/letter columns
            if col.lower() in ['bonus', 'letter']:
                if data[col].apply(lambda x: isinstance(x, str) and len(str(x).strip()) == 1 if pd.notna(x) else False).any():
                    letter_cols.append(col)
                    continue
                    
            # Check for letter/bonus column based on content
            if data[col].apply(lambda x: isinstance(x, str) and len(str(x).strip()) <= 2 and 
                               str(x).strip().isalpha() if pd.notna(x) else False).any():
                letter_cols.append(col)
                continue
                
            # Try converting to numeric
            temp = pd.to_numeric(data[col], errors='coerce')
            non_numeric = temp.isna() & data[col].notna()
            if non_numeric.any():
                stream_messages.append(
                    f"Column '{col}' has {non_numeric.sum()} non-numeric values. Examples: "
                    f"{data.loc[non_numeric, col].head().tolist()}"
                )
                logger.warning(f"Column '{col}' has {non_numeric.sum()} non-numeric values")
            if temp.notna().sum() > 0:  # At least some valid numeric values
                valid_numeric_cols.append(col)

        stream_messages.append(f"\nDate columns: {date_cols}")
        stream_messages.append(f"Numeric columns: {len(valid_numeric_cols)} ({valid_numeric_cols})")
        stream_messages.append(f"Letter columns: {len(letter_cols)} ({letter_cols})")
        logger.info(f"Detected {len(date_cols)} date, {len(valid_numeric_cols)} numeric, {len(letter_cols)} letter columns")

        # If no letter column was found, check the last column in each numeric sequence
        # Sometimes the Bonus column might be a single letter without a header
        if len(letter_cols) == 0:
            for i, col in enumerate(valid_numeric_cols):
                if i > 0 and i < len(valid_numeric_cols) - 1:
                    next_col = valid_numeric_cols[i+1]
                    if data[col].notna().sum() > data[next_col].notna().sum() * 2:  # Significant drop-off
                        # Check if the next column has letters
                        if data[next_col].apply(lambda x: isinstance(x, str) and str(x).strip().isalpha() 
                                             if pd.notna(x) else False).any():
                            letter_cols.append(next_col)
                            valid_numeric_cols.remove(next_col)
                            break

        if len(date_cols) == 0:
            error_msg = "Error: No date column detected. Please ensure your file has at least one column with dates."
            stream_messages.append(error_msg)
            logger.error(error_msg)
            return render_template("index.html")

        if len(valid_numeric_cols) < 1:
            error_msg = "Error: No numeric columns detected. Please ensure your file has at least one column with numbers."
            stream_messages.append(error_msg)
            logger.error(error_msg)
            return render_template("index.html")

        # Use the first date column
        date_col = date_cols[0]
        
        # If no letter column was found, we'll use the day of week as the target variable
        if len(letter_cols) == 0:
            stream_messages.append("No letter column found. Will use day of week as target variable.")
            logger.warning("No letter column found. Using day of week as target variable.")
        else:
            letter_col = letter_cols[0]
        
        num_cols = valid_numeric_cols

        # Parse dates with improved parsing
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
                
            date_str = str(date_str).strip()
            
            # Try pandas' default parser first
            try:
                return pd.to_datetime(date_str)
            except:
                pass
                
            # Clean up the date string
            # Remove day name if it appears at the end
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in day_names:
                if date_str.endswith(day):
                    date_str = date_str[:-len(day)].strip()
                if date_str.startswith(day):
                    date_str = date_str[len(day):].strip()
            
            # Remove commas and extra spaces
            date_str = re.sub(r'\s+', ' ', date_str.replace(',', ' '))
            
            # Handle "YYYY-Month-DD Day" format
            match = re.match(r'(\d{4})-([A-Za-z]+)-(\d+)', date_str)
            if match:
                year, month, day = match.groups()
                try:
                    month_num = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
                        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }[month]
                    return pd.Timestamp(int(year), month_num, int(day))
                except:
                    pass
            
            # Handle "Day Month DD, YYYY" format (e.g., "Monday May 05, 2025")
            match = re.match(r'(?:[A-Za-z]+ )?([A-Za-z]+) (\d+)(?:,)? (\d{4})', date_str)
            if match:
                month, day, year = match.groups()
                try:
                    month_num = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
                        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }[month]
                    return pd.Timestamp(int(year), month_num, int(day))
                except:
                    pass
            
            # Try common date formats
            formats = [
                '%Y-%b-%d', '%Y-%m-%d', '%B %d %Y', '%b %d %Y', '%d %b %Y', '%d %B %Y',
                '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y'
            ]
            
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
                    
            # If all else fails, try a more aggressive approach
            # Extract year, month, day if possible
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year = int(year_match.group(0))
                
                # Try to find month
                month_match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\b', date_str, re.IGNORECASE)
                if month_match:
                    month = month_match.group(0).capitalize()
                    month_num = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
                        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }.get(month, None)
                    
                    if month_num:
                        # Find day - look for 1-2 digits that aren't part of the year
                        day_match = re.search(r'\b(\d{1,2})\b', re.sub(r'\b\d{4}\b', '', date_str))
                        if day_match:
                            day = int(day_match.group(0))
                            if 1 <= day <= 31:
                                try:
                                    return pd.Timestamp(year, month_num, day)
                                except:
                                    pass
            
            # If we still can't parse, return NaT
            return pd.NaT

        data[date_col] = data[date_col].apply(parse_date)
        
        # Check if date parsing was successful
        if data[date_col].isna().all():
            error_msg = "Error: Could not parse any dates in the date column"
            stream_messages.append(error_msg)
            logger.error(error_msg)
            return render_template("index.html")
            
        # Show date range
        min_date = data[date_col].min()
        max_date = data[date_col].max()
        stream_messages.append(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        logger.info(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Drop rows with invalid dates
        data = data.dropna(subset=[date_col])
        stream_messages.append(f"After date cleaning, {len(data)} rows remain")
        logger.info(f"After date cleaning, {len(data)} rows remain")

        # Clean numeric columns
        for col in num_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isna().all():
                stream_messages.append(f"Warning: Column '{col}' contains no valid numeric data")
                logger.warning(f"Column '{col}' contains no valid numeric data")
                num_cols.remove(col)

        if not num_cols:
            error_msg = "Error: No valid numeric columns remain after cleaning"
            stream_messages.append(error_msg)
            logger.error(error_msg)
            return render_template("index.html")

        # Drop rows with NaN values in numeric columns
        data = data.dropna(subset=num_cols)
        stream_messages.append(f"After numeric cleaning, {len(data)} rows remain")
        logger.info(f"After numeric cleaning, {len(data)} rows remain")

        if len(data) == 0:
            error_msg = "Error: No valid data remains after cleaning. Check for missing or non-numeric values."
            stream_messages.append(error_msg)
            logger.error(error_msg)
            return render_template("index.html")

        # Extract day of the week
        data['Day'] = data[date_col].dt.day_name()
        days = data['Day'].dropna().unique().tolist()
        stream_messages.append(f"Days of the week: {days}")
        logger.info(f"Extracted days: {days}")
        
        # If no letter column was found, use the day of week as target
        if len(letter_cols) == 0:
            letter_col = 'Day'
            letter_cols = ['Day']

        # Feature engineering
        day_dummies = pd.get_dummies(data['Day'], prefix='day')
        data = pd.concat([data, day_dummies], axis=1)

        for col in num_cols:
            data[f'{col}_mod10'] = data[col] % 10
            data[f'{col}_div10'] = data[col] // 10

        data['sum_numbers'] = data[num_cols].sum(axis=1)
        data['max_diff'] = data[num_cols].max(axis=1) - data[num_cols].min(axis=1)
        
        # Add month features
        data['Month'] = data[date_col].dt.month
        data['DayOfMonth'] = data[date_col].dt.day
        data['DayOfYear'] = data[date_col].dt.dayofyear
        
        # Add time-based features for periodic patterns
        data['Month_sin'] = np.sin(2 * np.pi * data['Month']/12)
        data['Month_cos'] = np.cos(2 * np.pi * data['Month']/12)
        data['DayOfWeek_sin'] = np.sin(2 * np.pi * data[date_col].dt.dayofweek/7)
        data['DayOfWeek_cos'] = np.cos(2 * np.pi * data[date_col].dt.dayofweek/7)
        
        # Create lag features for each numeric column (previous week's numbers)
        data = data.sort_values(by=date_col)
        for col in num_cols:
            data[f'{col}_prev'] = data[col].shift(1)
            data[f'{col}_prev_week'] = data[col].shift(7)
            
        # Add parity features (odd/even count)
        data['odd_count'] = data[num_cols].apply(lambda x: sum(val % 2 == 1 for val in x), axis=1)
        data['even_count'] = data[num_cols].apply(lambda x: sum(val % 2 == 0 for val in x), axis=1)

        # Create full list of feature columns
        feature_cols = (
            list(num_cols) + 
            [f'{col}_mod10' for col in num_cols] + 
            [f'{col}_div10' for col in num_cols] + 
            ['sum_numbers', 'max_diff', 'Month', 'DayOfMonth', 
             'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
             'odd_count', 'even_count'] + 
            list(day_dummies.columns) +
            [f'{col}_prev' for col in num_cols if f'{col}_prev' in data.columns] +
            [f'{col}_prev_week' for col in num_cols if f'{col}_prev_week' in data.columns]
        )
        
        # Drop rows with NaN in feature columns (from lag features)
        data = data.dropna(subset=feature_cols)
        
        stream_messages.append(f"Using {len(feature_cols)} features")
        logger.info(f"Created {len(feature_cols)} features")

        # Prepare data
        X = data[feature_cols]
        y = data[letter_col]
        y_numeric, y_labels = pd.factorize(y)
        stream_messages.append(f"Predicting target: {set(y)}")
        logger.info(f"Predicting target: {set(y)}")

        # Find number range
        min_val = min(data[num_cols].min())
        max_val = max(data[num_cols].max())
        stream_messages.append(f"Number range: {min_val} to {max_val}")
        logger.info(f"Number range: {min_val} to {max_val}")

        # Evaluate models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)
        }

        tscv = TimeSeriesSplit(n_splits=min(5, len(data) // 5))  # Adapt splits to data size
        best_model_name, best_score = None, 0

        stream_messages.append("\nEvaluating models...")
        logger.info("Starting model evaluation")
        for name, model in models.items():
            all_preds, all_true = [], []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_numeric[train_idx], y_numeric[test_idx]
                model.fit(X_train, y_train)
                preds = np.round(model.predict(X_test)).astype(int)
                preds = np.clip(preds, 0, len(y_labels)-1)
                all_preds.extend(preds)
                all_true.extend(y_test)
            accuracy = accuracy_score(all_true, all_preds)
            stream_messages.append(f"{name} accuracy: {accuracy:.2%}")
            logger.info(f"{name} accuracy: {accuracy:.2%}")
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name

        stream_messages.append(f"Best model: {best_model_name} with {best_score:.2%} accuracy")
        logger.info(f"Best model: {best_model_name} with {best_score:.2%} accuracy")
        best_model = models[best_model_name]
        best_model.fit(X, y_numeric)

        # Make predictions for next week
        last_date = data[date_col].max()
        if pd.isna(last_date):
            last_date = datetime.datetime.now()
            stream_messages.append("Warning: No valid last date found, using current date")
            logger.warning("No valid last date found, using current date")

        stream_messages.append("\n=== NEXT WEEK PREDICTIONS ===")
        logger.info("Generating next week predictions")
        days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Get the most recent row for features like previous week values
        last_row = data.sort_values(by=date_col).iloc[-1]
        
        for i in range(1, 8):
            next_date = last_date + datetime.timedelta(days=i)
            day_name = days_list[next_date.weekday()]
            day_data = data[data['Day'] == day_name]
            
            # Base numbers can come from day-specific averages if we have enough data
            base_numbers = []
            for col in num_cols:
                if len(day_data) >= 3:  # We have enough day-specific data
                    base = day_data[col].mean()
                else:  # Not enough day data, use overall average
                    base = data[col].mean()
                base_numbers.append(round(base))
            
            # Create prediction row with all features
            pred_row = pd.DataFrame({col: [base_numbers[i]] for i, col in enumerate(num_cols)})
            
            # Add derived features
            for col in num_cols:
                pred_row[f'{col}_mod10'] = pred_row[col] % 10
                pred_row[f'{col}_div10'] = pred_row[col] // 10
                
                # Add lag features from the most recent data
                if f'{col}_prev' in X.columns:
                    pred_row[f'{col}_prev'] = last_row[col]
                if f'{col}_prev_week' in X.columns:
                    pred_row[f'{col}_prev_week'] = data[
                        data[date_col] == (next_date - datetime.timedelta(days=7))
                    ][col].mean() if not data[
                        data[date_col] == (next_date - datetime.timedelta(days=7))
                    ].empty else last_row[col]
            
            # Add sum and difference
            pred_row['sum_numbers'] = pred_row[num_cols].sum(axis=1)
            pred_row['max_diff'] = pred_row[num_cols].max(axis=1) - pred_row[num_cols].min(axis=1)
            
            # Add time features
            pred_row['Month'] = next_date.month
            pred_row['DayOfMonth'] = next_date.day
            pred_row['DayOfYear'] = next_date.timetuple().tm_yday
            pred_row['Month_sin'] = np.sin(2 * np.pi * next_date.month/12)
            pred_row['Month_cos'] = np.cos(2 * np.pi * next_date.month/12)
            pred_row['DayOfWeek_sin'] = np.sin(2 * np.pi * next_date.weekday()/7)
            pred_row['DayOfWeek_cos'] = np.cos(2 * np.pi * next_date.weekday()/7)
            
            # Add parity features
            pred_row['odd_count'] = pred_row[num_cols].apply(lambda x: sum(val % 2 == 1 for val in x), axis=1)
            pred_row['even_count'] = pred_row[num_cols].apply(lambda x: sum(val % 2 == 0 for val in x), axis=1)
            
            # Add day dummies
            for day in days_list:
                pred_row[f'day_{day}'] = 1 if day == day_name else 0
            
            # Make sure we have all feature columns
            for col in set(feature_cols) - set(pred_row.columns):
                pred_row[col] = 0
                
            # Select only the features used in training
            pred_row = pred_row[feature_cols]

            # Make prediction
            pred_numeric = int(round(best_model.predict(pred_row)[0]))
            pred_numeric = min(max(0, pred_numeric), len(y_labels)-1)
            predicted_letter = y_labels[pred_numeric]
            confidence = best_score

            # Generate refined numbers with historical patterns
            refined_numbers = []
            for col in num_cols:
                # Check for frequent numbers on this day
                frequent_nums = []
                if len(day_data) >= 3:
                    frequent_nums = day_data[col].value_counts().index.tolist()[:3]
                
                if frequent_nums and np.random.random() < 0.6:
                    number = int(np.random.choice(frequent_nums))
                else:
                    # Use mean with variation
                    base = data[col].mean()
                    variation = np.random.normal(0, data[col].std() / 3)
                    number = int(round(base + variation))
                    number = max(int(min_val), min(int(max_val), number))
                refined_numbers.append(number)

            stream_messages.append(
                f"{day_name} ({next_date.strftime('%Y-%m-%d')}): {refined_numbers} → "
                f"{predicted_letter} (Confidence: {confidence:.2%})"
            )
            logger.info(f"Prediction for {day_name}: {refined_numbers} → {predicted_letter}")

        # Generate top 5 predictions
        stream_messages.append("\n=== TOP 5 HIGHEST CONFIDENCE PREDICTIONS ===")
        logger.info("Generating top 5 predictions")
        n_samples = 1000
        test_data = pd.DataFrame({
            col: [randint(int(min_val), int(max_val)) for _ in range(n_samples)]
            for col in num_cols
        })
        
        # Add derived features to test data
        for col in num_cols:
            test_data[f'{col}_mod10'] = test_data[col] % 10
            test_data[f'{col}_div10'] = test_data[col] // 10
        test_data['sum_numbers'] = test_data[num_cols].sum(axis=1)
        test_data['max_diff'] = test_data[num_cols].max(axis=1) - test_data[num_cols].min(axis=1)
        test_data['odd_count'] = test_data[num_cols].apply(lambda x: sum(val % 2 == 1 for val in x), axis=1)
        test_data['even_count'] = test_data[num_cols].apply(lambda x: sum(val % 2 == 0 for val in x), axis=1)

        top_predictions = []
        for day in days_list:
            # Create a copy of test data for this day
            day_test_data = test_data.copy()
            
            # Add day-specific features
            next_day_date = last_date + datetime.timedelta(days=days_list.index(day) + 1)
            day_test_data['Month'] = next_day_date.month
            day_test_data['DayOfMonth'] = next_day_date.day
            day_test_data['DayOfYear'] = next_day_date.timetuple().tm_yday
            day_test_data['Month_sin'] = np.sin(2 * np.pi * next_day_date.month/12)
            day_test_data['Month_cos'] = np.cos(2 * np.pi * next_day_date.month/12)
            day_test_data['DayOfWeek_sin'] = np.sin(2 * np.pi * next_day_date.weekday()/7)
            day_test_data['DayOfWeek_cos'] = np.cos(2 * np.pi * next_day_date.weekday()/7)
            
            # Add lag features
            for col in num_cols:
                if f'{col}_prev' in feature_cols:
                    day_test_data[f'{col}_prev'] = last_row[col]
                if f'{col}_prev_week' in feature_cols:
                    day_test_data[f'{col}_prev_week'] = data[
                        data[date_col] == (next_day_date - datetime.timedelta(days=7))
                    ][col].mean() if not data[
                        data[date_col] == (next_day_date - datetime.timedelta(days=7))
                    ].empty else last_row[col]
            
            # Add day dummies
            for d in days_list:
                day_test_data[f'day_{d}'] = 1 if d == day else 0
                
            # Fill in any missing features
            for col in set(feature_cols) - set(day_test_data.columns):
                day_test_data[col] = 0
                
            # Make sure we have only the features used in training
            day_test_data = day_test_data[feature_cols]
            
            # Make predictions
            predictions = best_model.predict(day_test_data)
            confidences = [1 - min(1, abs(p - round(p)) / 2) for p in predictions]
            day_test_data['confidence'] = confidences
            day_test_data['predicted_letter'] = [
                y_labels[int(round(p)) % len(y_labels)] for p in predictions
            ]
            
            # Find best prediction for this day
            best_idx = day_test_data['confidence'].argmax()
            top_predictions.append({
                'day': day,
                'numbers': day_test_data.iloc[best_idx][num_cols].tolist(),
                'letter': day_test_data.iloc[best_idx]['predicted_letter'],
                'confidence': day_test_data.iloc[best_idx]['confidence']
            })

        for pred in sorted(top_predictions, key=lambda x: x['confidence'], reverse=True)[:5]:
            stream_messages.append(
                f"{pred['day']}: {[int(n) for n in pred['numbers']]} → {pred['letter']} "
                f"(Confidence: {pred['confidence']:.2%})"
            )
            logger.info(f"Top prediction for {pred['day']}: {pred['numbers']} → {pred['letter']}")

        # After the "TOP 5 HIGHEST CONFIDENCE PREDICTIONS" section, add:

        # Generate final recommendations
        stream_messages.append("\n=== FINAL RECOMMENDATIONS ===")
        logger.info("Generating final recommendations")
        
        # Calculate overall best prediction
        best_prediction = sorted(top_predictions, key=lambda x: x['confidence'], reverse=True)[0]
        
        stream_messages.append("Based on comprehensive statistical analysis, here are the most reliable predictions:")
        
        # Get the most recent date in the dataset
        last_date = data[date_col].max()
        last_draw = data.loc[data[date_col] == last_date]
        
        if not last_draw.empty:
            stream_messages.append(f"Last draw ({last_date.strftime('%Y-%b-%d %A')}): {last_draw[num_cols].values[0].tolist()} → {last_draw[letter_col].values[0]}")
        
        # Create day-specific predictions with alternatives
        for day in days_list:
            next_date = last_date + datetime.timedelta(days=days_list.index(day) + 1)
            
            # Find prediction for this day
            day_pred = next(pred for pred in top_predictions if pred['day'] == day)
            
            # Get most frequent numbers for this day
            day_data = data[data['Day'] == day]
            frequent_numbers = []
            for col in num_cols:
                if not day_data.empty:
                    freq_nums = day_data[col].value_counts().head(3).index.tolist()
                    frequent_numbers.extend(freq_nums)
                else:
                    freq_nums = data[col].value_counts().head(3).index.tolist()
                    frequent_numbers.extend(freq_nums)
            
            # Count frequency of each number
            num_freq = Counter(frequent_numbers)
            most_frequent = [num for num, _ in num_freq.most_common(len(num_cols))]
            
            # Ensure we have enough numbers
            while len(most_frequent) < len(num_cols):
                most_frequent.append(int(data[num_cols[0]].mean()))
            
            # Predict letter for these frequent numbers
            freq_test_data = pd.DataFrame({col: [most_frequent[i]] for i, col in enumerate(num_cols)})
            
            # Add derived features
            for col in num_cols:
                freq_test_data[f'{col}_mod10'] = freq_test_data[col] % 10
                freq_test_data[f'{col}_div10'] = freq_test_data[col] // 10
            
            freq_test_data['sum_numbers'] = freq_test_data[num_cols].sum(axis=1)
            freq_test_data['max_diff'] = freq_test_data[num_cols].max(axis=1) - freq_test_data[num_cols].min(axis=1)
            freq_test_data['odd_count'] = freq_test_data[num_cols].apply(lambda x: sum(val % 2 == 1 for val in x), axis=1)
            freq_test_data['even_count'] = freq_test_data[num_cols].apply(lambda x: sum(val % 2 == 0 for val in x), axis=1)
            
            # Add time features
            freq_test_data['Month'] = next_date.month
            freq_test_data['DayOfMonth'] = next_date.day
            freq_test_data['DayOfYear'] = next_date.timetuple().tm_yday
            freq_test_data['Month_sin'] = np.sin(2 * np.pi * next_date.month/12)
            freq_test_data['Month_cos'] = np.cos(2 * np.pi * next_date.month/12)
            freq_test_data['DayOfWeek_sin'] = np.sin(2 * np.pi * next_date.weekday()/7)
            freq_test_data['DayOfWeek_cos'] = np.cos(2 * np.pi * next_date.weekday()/7)
            
            # Add day dummies
            for d in days_list:
                freq_test_data[f'day_{d}'] = 1 if d == day else 0
            
            # Add lag features
            for col in num_cols:
                if f'{col}_prev' in feature_cols:
                    freq_test_data[f'{col}_prev'] = last_row[col]
                if f'{col}_prev_week' in feature_cols:
                    freq_test_data[f'{col}_prev_week'] = data[
                        data[date_col] == (next_date - datetime.timedelta(days=7))
                    ][col].mean() if not data[
                        data[date_col] == (next_date - datetime.timedelta(days=7))
                    ].empty else last_row[col]
            
            # Fill in any missing features
            for col in set(feature_cols) - set(freq_test_data.columns):
                freq_test_data[col] = 0
            
            # Make sure we have only the features used in training
            freq_test_data = freq_test_data[feature_cols]
            
            # Predict letter for frequent numbers
            freq_pred = best_model.predict(freq_test_data)[0]
            freq_pred_numeric = int(round(freq_pred))
            freq_pred_numeric = min(max(0, freq_pred_numeric), len(y_labels)-1)
            freq_pred_letter = y_labels[freq_pred_numeric]
            freq_confidence = 1 - min(1, abs(freq_pred - round(freq_pred)) / 2)
            
            # Output both predictions
            stream_messages.append(f"\n{next_date.strftime('%Y-%b-%d %A')}:")
            stream_messages.append(f"Best prediction: Numbers {[int(n) for n in day_pred['numbers']]} → Letter {day_pred['letter']} (Confidence: {day_pred['confidence']:.1%})")
            stream_messages.append(f"Alternative prediction using most frequent numbers:")
            stream_messages.append(f"Numbers {most_frequent[:len(num_cols)]} → Letter {freq_pred_letter} (Confidence: {freq_confidence:.1%})")
        
        # Find overall best prediction
        stream_messages.append("\n=== OVERALL BEST PREDICTION ===")
        best_day_pred = sorted(top_predictions, key=lambda x: x['confidence'], reverse=True)[0]
        stream_messages.append(f"The most confident prediction for the week is for {best_day_pred['day']}:")
        stream_messages.append(f"Numbers: {[int(n) for n in best_day_pred['numbers']]}")
        stream_messages.append(f"Letter: {best_day_pred['letter']}")
        stream_messages.append(f"Confidence: {best_day_pred['confidence']:.1%}")
        
        # Count all predicted numbers to find hot numbers
        all_predicted_numbers = []
        for pred in top_predictions:
            all_predicted_numbers.extend([int(n) for n in pred['numbers']])
        
        # Find hot numbers (most frequently predicted)
        hot_numbers = Counter(all_predicted_numbers).most_common(10)
        
        stream_messages.append("\n=== HOT NUMBERS STRATEGY ===")
        stream_messages.append("Overall hottest numbers for this week:")
        for num, count in hot_numbers:
            stream_messages.append(f"Number {num}: {count} occurrences")
        
        # Find cold numbers (historically common but not in predictions)
        historical_numbers = []
        for _, row in data.iterrows():
            historical_numbers.extend([int(row[col]) for col in num_cols])
        
        historical_freq = Counter(historical_numbers)
        predicted_set = set(all_predicted_numbers)
        
        cold_common_numbers = [(num, count) for num, count in historical_freq.most_common(50) 
                             if num not in predicted_set and count > len(data) / 10][:10]
        
        stream_messages.append("\n=== COLD NUMBERS STRATEGY ===")
        stream_messages.append("Cold but historically common numbers (contrarian strategy):")
        for num, count in cold_common_numbers:
            stream_messages.append(f"Number {num}: {count} historical occurrences")
        
        # Calculate win probabilities by day
        day_counts = data['Day'].value_counts()
        total_days = day_counts.sum()
        day_probs = {day: count/total_days for day, count in day_counts.items()}
        
        # Sort days by probability
        sorted_days = sorted(day_probs.items(), key=lambda x: x[1], reverse=True)
        
        stream_messages.append("\n=== DAYS WITH HIGHEST WIN PROBABILITY ===")
        for day, prob in sorted_days:
            stream_messages.append(f"{day}: {prob:.4f} ({prob:.1%})")
        
        # Calculate correlation between numbers
        num_corr = data[num_cols].corr()
        strong_corr = []
        
        for i, col1 in enumerate(num_cols):
            for j, col2 in enumerate(num_cols):
                if i < j and abs(num_corr.loc[col1, col2]) > 0.3:  # Threshold for strong correlation
                    strong_corr.append((col1, col2, num_corr.loc[col1, col2]))
        
        if strong_corr:
            stream_messages.append("\n=== CORRELATED NUMBERS ===")
            stream_messages.append("These numbers tend to appear together:")
            for col1, col2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                stream_messages.append(f"{col1} and {col2}: {corr:.2f} correlation")
        
        # Find pattern in letter sequence if letters are not days
        if letter_col != 'Day' and len(set(data[letter_col])) < 26:
            letter_seq = data.sort_values(by=date_col)[letter_col].tolist()[-10:]
            stream_messages.append("\n=== LETTER SEQUENCE PATTERN ===")
            stream_messages.append(f"Last 10 letters: {letter_seq}")
            
            # Try to identify patterns in letter sequence
            repeats = []
            for i in range(1, 6):  # Check for patterns of length 1 to 5
                if len(letter_seq) >= i*2:
                    if letter_seq[-i:] == letter_seq[-2*i:-i]:
                        repeats.append(i)
            
            if repeats:
                stream_messages.append(f"Possible repeating pattern of length {repeats[0]} detected")
                next_letters = letter_seq[-repeats[0]:]
                stream_messages.append(f"If pattern continues, next letters would be: {next_letters}")
        
        # Final prediction summary
        stream_messages.append("\n=== FINAL PREDICTION SUMMARY ===")
        stream_messages.append(f"Prediction for {next_date.strftime('%Y-%b-%d %A')}: {[int(n) for n in best_day_pred['numbers']]}")
        stream_messages.append(f"Letter: {best_day_pred['letter']} (Confidence: {best_day_pred['confidence']:.1%})")
        
        # Check model reliability
        if best_score >= 0.7:
            stream_messages.append("\n✓ Model has high reliability (>70% accuracy)")
        elif best_score >= 0.6:
            stream_messages.append("\n✓ Model meets the standard reliability threshold (>60% accuracy)")
        else:
            stream_messages.append("\n⚠ Model has lower than optimal reliability")
            stream_messages.append("Recommendations to improve predictions:")
            stream_messages.append("1. Collect more historical data")
            stream_messages.append("2. Use the alternative prediction based on frequent numbers")
            stream_messages.append("3. Consider the day with highest win probability")

        stream_messages.append(f"\nOverall model accuracy: {best_score:.2%}")
        logger.info(f"Overall model accuracy: {best_score:.2%}")
        stream_messages.append(
            "✓ Model meets the 60% accuracy threshold" if best_score >= 0.6 else
            "✗ Model does not meet the 60% accuracy threshold\n"
            "Recommendations:\n1. Collect more data\n2. Add time-based patterns\n3. Include sum frequency"
        )

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
        stream_messages.append(error_msg)
        logger.error(error_msg)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)