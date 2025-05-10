# app.py
import os
import json
import logging
import redis
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from celery import Celery
from dateutil.parser import parse as parse_date
from dotenv import load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_file, url_for)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration from environment variables with defaults
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', os.urandom(24)),
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    MAX_CONTENT_LENGTH=int(os.getenv('MAX_FILE_SIZE', 16)) * 1024 * 1024,  # 16MB default
    ALLOWED_EXTENSIONS={'csv', 'xlsx', 'xls', 'json'},
    REDIS_URL=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    CELERY_BROKER_URL=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    RATE_LIMIT=os.getenv('RATE_LIMIT', '100 per day, 10 per minute'),
    ASYNC_THRESHOLD=int(os.getenv('ASYNC_THRESHOLD', 5 * 1024 * 1024))  # 5MB
)

# Initialize Celery
celery = Celery(
    app.name,
    broker=app.config['CELERY_BROKER_URL'],
    backend=app.config['CELERY_RESULT_BACKEND']
)

# Rate limiting setup
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=app.config['REDIS_URL'],
    default_limits=[app.config['RATE_LIMIT']]
)

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True, parents=True)


class DataFormat(Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"


class DataCleaner:
    """Professional data cleaning and preprocessing system with all requested features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.config = config or {
            "outlier_contamination": 0.01,
            "impute_strategy": "knn",
            "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%d-%b-%y"],
            "text_columns": None,
            "numeric_columns": None,
            "categorical_columns": None,  # New: explicitly specify categorical columns
            "protect_columns": ["id"],
            "drop_threshold": 0.5
        }
        self._validate_config()
    def _validate_config(self):
        """Validate configuration parameters."""
        contamination = self.config.get("outlier_contamination", 0.01)
        if not isinstance(contamination, (int, float)) or not 0 <= contamination <= 0.5:
            raise ValueError("Contamination should be a number between 0 and 0.5")

        valid_strategies = ["mean", "median", "knn", "drop"]
        if self.config.get("impute_strategy") not in valid_strategies:
            raise ValueError(f"Invalid impute strategy. Use one of {valid_strategies}")

        drop_threshold = self.config.get("drop_threshold", 0.5)
        if not isinstance(drop_threshold, (int, float)) or not 0 <= drop_threshold <= 1:
            raise ValueError("Drop threshold should be between 0 and 1")
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various formats with robust error handling."""
        file_path = Path(file_path)
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix.lower() in ('.xlsx', '.xls'):
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def _get_categorical_columns(self, df: pd.DataFrame) -> list:
        """Get categorical columns, either from config or auto-detected."""
        if df.empty:
            return []

        if self.config.get("categorical_columns"):
            return [col for col in self.config["categorical_columns"] if col in df.columns]

        # Auto-detect: text columns with less than 50 unique values
        text_cols = self._get_text_columns(df)
        return [col for col in text_cols
                if len(df[col].dropna().unique()) < 50 and len(df[col].dropna().unique()) > 1]

    def _get_categorical_columns(self, df: pd.DataFrame) -> list:
        """Get categorical columns, either from config or auto-detected."""
        if df.empty:
            return []

        if self.config.get("categorical_columns"):
            return [col for col in self.config["categorical_columns"] if col in df.columns]

        # Auto-detect: text columns with less than 50 unique values
        text_cols = self._get_text_columns(df)
        return [col for col in text_cols
                if len(df[col].dropna().unique()) < 50 and len(df[col].dropna().unique()) > 1]

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with mode imputation for categorical data."""
        if df.empty:
            return df

        strategy = self.config.get("impute_strategy", "knn")
        protected = self.config.get("protect_columns", [])
        drop_threshold = self.config.get("drop_threshold", 0.5)

        if strategy == "drop":
            threshold = int(len(df.columns) * drop_threshold)
            df_clean = df.dropna(thresh=threshold)
            logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
            return df_clean

        df = df.copy()

        # 1. First handle categorical columns with mode imputation
        categorical_cols = [col for col in self._get_categorical_columns(df)
                            if col not in protected]

        for col in categorical_cols:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')

        # 2. Then handle numeric columns based on strategy
        numeric_cols = [col for col in self._get_numeric_columns(df)
                        if col not in protected and col not in categorical_cols]

        if strategy == "mean" and numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "median" and numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "knn" and numeric_cols:
            try:
                imputer = KNNImputer(n_neighbors=3)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            except Exception:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # 3. Finally handle remaining text columns (non-categorical)
        text_cols = [col for col in self._get_text_columns(df)
                     if col not in protected and col not in categorical_cols]

        for col in text_cols:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

        return df

    def detect_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """More lenient outlier detection using Isolation Forest."""
        numeric_cols = [col for col in self._get_numeric_columns(df)
                        if col not in self.config.get("protect_columns", [])]

        if not numeric_cols:
            return df, pd.DataFrame()

        try:
            contamination = min(self.config.get("outlier_contamination", 0.01), 0.1)
            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                behaviour='new'
            )

            # Convert to numeric, handling errors
            numeric_data = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            outliers = clf.fit_predict(numeric_data.fillna(numeric_data.median()))
            outlier_mask = outliers == -1

            # Replace outliers with median instead of removing
            clean_df = df.copy()
            for col in numeric_cols:
                col_median = numeric_data[col].median()
                clean_df[col] = np.where(outlier_mask, col_median, df[col])

            outliers_df = df[outlier_mask]
            logger.info(f"Adjusted {len(outliers_df)} outlier values")
            return clean_df, outliers_df

        except Exception as e:
            logger.error(f"Outlier detection failed: {str(e)}")
            return df, pd.DataFrame()

    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """More forgiving data standardization."""
        if df.empty:
            return df

        df = df.copy()
        protected = self.config.get("protect_columns", [])

        # Text columns
        text_cols = [col for col in self._get_text_columns(df)
                     if col not in protected]
        for col in text_cols:
            df[col] = df[col].astype(str).str.lower().str.strip()

        # Date columns
        date_cols = []
        for col in df.columns:
            if col in protected or col in self._get_numeric_columns(df):
                continue
            try:
                if self._is_date_column(df[col]):
                    date_cols.append(col)
            except Exception:
                continue

        for col in date_cols:
            df[col] = df[col].apply(self._safe_parse_date)

        return df

    def _safe_parse_date(self, val: Any) -> Any:
        """Parse dates while preserving original value on failure."""
        if pd.isna(val):
            return val

        try:
            parsed = parse_date(str(val), fuzzy=True)
            return parsed.strftime('%Y-%m-%d') if parsed else val
        except Exception:
            return val

    def _is_date_column(self, series: pd.Series) -> bool:
        """More conservative date detection."""
        if is_numeric_dtype(series):
            return False

        sample = series.dropna().sample(min(5, len(series))) if not series.empty else pd.Series()
        if sample.empty:
            return False

        date_count = 0
        for val in sample:
            try:
                if pd.notna(val) and any(sep in str(val) for sep in ['-', '/', '.']):
                    if self._safe_parse_date(val) != val:
                        date_count += 1
            except Exception:
                continue

        return (date_count / len(sample)) > 0.6  # 60% threshold

    def _get_numeric_columns(self, df: pd.DataFrame) -> list:
        """Get numeric columns, either from config or auto-detected."""
        if df.empty:
            return []

        if self.config.get("numeric_columns"):
            return [col for col in self.config["numeric_columns"] if col in df.columns]
        return df.select_dtypes(include=np.number).columns.tolist()

    def _get_text_columns(self, df: pd.DataFrame) -> list:
        """Get text columns, either from config or auto-detected."""
        if df.empty:
            return []

        if self.config.get("text_columns"):
            return [col for col in self.config["text_columns"] if col in df.columns]
        return df.select_dtypes(include=['object']).columns.tolist()

    def process_pipeline(self, input_path: Union[str, Path],
                         output_path: Union[str, Path],
                         output_format: Union[DataFormat, str] = DataFormat.CSV) -> Dict[str, Any]:
        """Complete data processing pipeline with row preservation."""
        stats = {'original_rows': 0, 'rows_after_cleaning': 0, 'status': 'failed'}

        try:
            df = self.load_data(input_path)
            stats['original_rows'] = len(df)

            # Handle missing values (preserves all rows)
            df = self.handle_missing_values(df)

            # Detect and adjust outliers (preserves all rows)
            df, _ = self.detect_outliers(df)

            # Standardize data (preserves all rows)
            df = self.standardize_data(df)

            self.export_data(df, output_path, output_format)
            stats.update({
                'rows_after_cleaning': len(df),
                'status': 'success'
            })
            return stats

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            stats['error'] = str(e)
            return stats

    def export_data(self, df: pd.DataFrame, output_path: Union[str, Path],
                    format_type: Union[DataFormat, str]) -> None:
        """Export cleaned data."""
        try:
            if isinstance(format_type, str):
                format_type = DataFormat(format_type.lower())

            if format_type == DataFormat.CSV:
                df.to_csv(output_path, index=False)
            elif format_type == DataFormat.EXCEL:
                df.to_excel(output_path, index=False)
            elif format_type == DataFormat.JSON:
                df.to_json(output_path, orient='records', indent=2)

        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise

    # [Rest of your Flask routes and Celery tasks remain exactly the same]


@celery.task(bind=True)
def process_file_async(self, input_path, config, output_path):
    """Celery task for asynchronous file processing"""
    self.update_state(state='PROGRESS', meta={'status': 'Processing started'})

    cleaner = DataCleaner(config)

    try:
        result = cleaner.process_pipeline(
            input_path=input_path,
            output_path=output_path,
            output_format=config.get('output_format', 'csv')
        )

        if result['status'] == 'success':
            return os.path.basename(output_path)
        else:
            raise Exception(result.get('error', 'Processing failed'))
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-KEY') != os.getenv('API_KEY'):
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)

    return decorated_function


@app.route('/')
def landing():
    """Show the marketing landing page"""
    return render_template('landing.html')


@app.route('/clean', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def clean_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            file_size = os.path.getsize(upload_path)

            # Validate and sanitize contamination input
            try:
                contamination = float(request.form.get('contamination', 0.05))
                contamination = max(0.0, min(0.5, contamination))  # Ensure value is between 0 and 0.5
            except (ValueError, TypeError):
                contamination = 0.05  # Default value if parsing fails

            config = {
                "outlier_contamination": contamination,
                "impute_strategy": request.form.get('impute_strategy', 'knn'),
                "output_format": request.form.get('output_format', 'csv')
            }

            if file_size > app.config['ASYNC_THRESHOLD']:
                task = process_file_async.delay(
                    upload_path,
                    config,
                    os.path.join(app.config['UPLOAD_FOLDER'],
                                 f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
                )
                flash(f"Large file processing started. Task ID: {task.id}")
                return redirect(url_for('task_status', task_id=task.id))
            else:
                cleaner = DataCleaner(config)
                output_filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

                result = cleaner.process_pipeline(
                    input_path=upload_path,
                    output_path=output_path,
                    output_format=config['output_format']
                )

                if result['status'] == 'success':
                    return redirect(url_for('download', filename=output_filename))
                else:
                    flash(f"Processing failed: {result.get('error', 'Unknown error')}")
                    return redirect(request.url)

    return render_template('index.html')


@app.route('/api/clean', methods=['POST'])
@require_api_key
@limiter.limit("60 per minute")
def api_clean():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # Validate and sanitize contamination input
    if request.is_json:
        config = request.get_json()
        try:
            if 'outlier_contamination' in config:
                config['outlier_contamination'] = max(0.0, min(0.5, float(config['outlier_contamination'])))
        except (ValueError, TypeError):
            config['outlier_contamination'] = 0.05
    else:
        try:
            contamination = float(request.form.get('contamination', 0.05))
            contamination = max(0.0, min(0.5, contamination))
        except (ValueError, TypeError):
            contamination = 0.05

        config = {
            "outlier_contamination": contamination,
            "impute_strategy": request.form.get('impute_strategy', 'knn'),
            "output_format": request.form.get('output_format', 'csv')
        }

    file_size = os.path.getsize(upload_path)

    if file_size > app.config['ASYNC_THRESHOLD']:
        task = process_file_async.delay(
            upload_path,
            config,
            os.path.join(app.config['UPLOAD_FOLDER'],
                         f"cleaned_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        )
        return jsonify({
            "status": "processing",
            "task_id": task.id,
            "message": "Large file processing started"
        }), 202
    else:
        cleaner = DataCleaner(config)
        output_filename = f"cleaned_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        result = cleaner.process_pipeline(
            input_path=upload_path,
            output_path=output_path,
            output_format=config.get('output_format', 'csv')
        )

        try:
            os.remove(upload_path)
        except:
            pass

        if result['status'] == 'success':
            return send_file(
                output_path,
                as_attachment=True,
                download_name=output_filename
            )
        else:
            try:
                os.remove(output_path)
            except:
                pass
            return jsonify({"error": result.get('error', 'Processing failed')}), 500


@app.route('/tasks/<task_id>')
def task_status(task_id):
    task = process_file_async.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.info.get('result', None),
            'status': task.info.get('status', '')
        }
        if 'result' in response and response['result']:
            return redirect(url_for('download', filename=response['result']))
    else:
        response = {
            'state': task.state,
            'status': str(task.info),
        }
    return render_template('task_status.html', task=response)


@app.route('/download/<filename>')
def download(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/docs')
def api_docs():
    docs = {
        "description": "Data Cleaning API",
        "endpoints": {
            "/api/clean": {
                "method": "POST",
                "description": "Clean and process uploaded data file",
                "parameters": {
                    "file": "Data file to process (CSV, Excel, or JSON)",
                    "contamination": "Outlier detection sensitivity (0-0.5)",
                    "impute_strategy": "Missing value strategy (mean/median/knn/drop)",
                    "output_format": "Output format (csv/excel/json)"
                },
                "response": {
                    "success": "Returns cleaned data file",
                    "error": "Returns error message with status code"
                }
            }
        }
    }
    return jsonify(docs)


if __name__ == '__main__':
    app.run(debug=True)