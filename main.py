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
            "outlier_contamination": 0.05,
            "impute_strategy": "knn",
            "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"],
            "text_columns": None,
            "numeric_columns": None
        }
        # Ensure all expected keys exist
        self.config.setdefault("text_columns", None)
        self.config.setdefault("numeric_columns", None)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0 <= self.config.get("outlier_contamination", 0.05) <= 0.5:
            raise ValueError("Contamination should be between 0 and 0.5")

        valid_strategies = ["mean", "median", "knn", "drop"]
        if self.config.get("impute_strategy") not in valid_strategies:
            raise ValueError(f"Invalid impute strategy. Use one of {valid_strategies}")

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various formats with robust error handling."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix.lower() in ('.xlsx', '.xls'):
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.json_normalize(data)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            logger.info(f"Successfully loaded data from {file_path}. Columns: {df.columns.tolist()}")
            logger.info(f"Data types:\n{df.dtypes}")
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically handle missing values based on configuration."""
        if df.empty or df.isnull().sum().sum() == 0:
            logger.info("No missing values detected")
            return df

        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        strategy = self.config.get("impute_strategy", "knn")
        logger.info(f"Handling missing values using {strategy} strategy")

        numeric_cols = self._get_numeric_columns(df)
        text_cols = self._get_text_columns(df)

        if strategy == "drop":
            return df.dropna()
        elif strategy in ["mean", "median"] and numeric_cols:
            for col in numeric_cols:
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].median())
        elif strategy == "knn" and numeric_cols:
            try:
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            except ValueError as e:
                logger.warning(f"KNN imputation failed: {str(e)}. Falling back to mean imputation.")
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mean())

        # Handle text columns
        for col in text_cols:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

        return df

    def detect_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect outliers using machine learning approach."""
        numeric_cols = self._get_numeric_columns(df)
        if not numeric_cols or len(df) == 0:
            logger.warning("No numeric columns or empty dataframe for outlier detection")
            return df, pd.DataFrame()

        try:
            clf = IsolationForest(
                contamination=min(self.config.get("outlier_contamination", 0.05), 0.5),
                random_state=42
            )
            outliers = clf.fit_predict(df[numeric_cols])
            outlier_mask = outliers == -1

            clean_df = df[~outlier_mask]
            outliers_df = df[outlier_mask]

            logger.info(f"Detected {len(outliers_df)} outliers")
            return clean_df, outliers_df
        except Exception as e:
            logger.error(f"Outlier detection failed: {str(e)}")
            return df, pd.DataFrame()

    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats including text case and date formats."""
        if df.empty:
            return df

        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Standardize text columns to lowercase
        text_cols = self._get_text_columns(df)
        for col in text_cols:
            df[col] = df[col].astype(str).str.lower()

        # Standardize date columns - more careful handling
        date_cols = []
        for col in df.columns:
            try:
                if self._is_date_column(df[col]):
                    date_cols.append(col)
            except Exception as e:
                logger.warning(f"Could not check date format for column {col}: {str(e)}")
                continue

        for col in date_cols:
            try:
                df[col] = df[col].apply(lambda x: self._parse_date(str(x)) if pd.notna(x) else x)
            except Exception as e:
                logger.warning(f"Could not standardize dates in column {col}: {str(e)}")

        return df

    def _parse_date(self, date_val: Any) -> Optional[datetime]:
        """Parse date from various formats to standard format."""
        if pd.isna(date_val) or not str(date_val).strip():
            return None

        try:
            if isinstance(date_val, (datetime, pd.Timestamp)):
                return date_val
            return parse_date(str(date_val))
        except (ValueError, TypeError):
            # Skip logging for obviously non-date strings
            if not any(c.isdigit() for c in str(date_val)):
                return None
            logger.debug(f"Could not parse date: {date_val}")
            return None

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column contains date values."""
        if is_numeric_dtype(series):
            return False

        try:
            sample = series.dropna().sample(min(5, len(series))) if not series.empty else pd.Series()
            if sample.empty:
                return False

            date_count = 0
            for val in sample:
                try:
                    if pd.notna(val) and self._parse_date(str(val)) is not None:
                        date_count += 1
                except (ValueError, TypeError):
                    continue

            return (date_count / len(sample)) > 0.5  # Lower threshold to 50%
        except Exception as e:
            logger.warning(f"Date detection failed for column: {str(e)}")
            return False

    def _get_numeric_columns(self, df: pd.DataFrame) -> list:
        """Get numeric columns, either from config or auto-detected."""
        if not df.empty:
            if self.config.get("numeric_columns"):
                return [col for col in self.config["numeric_columns"] if col in df.columns]
            return df.select_dtypes(include=np.number).columns.tolist()
        return []

    def _get_text_columns(self, df: pd.DataFrame) -> list:
        """Get text columns, either from config or auto-detected."""
        if not df.empty:
            if self.config.get("text_columns"):
                return [col for col in self.config["text_columns"] if col in df.columns]
            return df.select_dtypes(include=['object']).columns.tolist()
        return []

    def export_data(self, df: pd.DataFrame, output_path: Union[str, Path],
                    format_type: Union[DataFormat, str]) -> None:
        """Export cleaned data in specified format."""
        output_path = Path(output_path)
        if isinstance(format_type, str):
            try:
                format_type = DataFormat(format_type.lower())
            except ValueError:
                raise ValueError(f"Invalid format type: {format_type}")

        try:
            if format_type == DataFormat.CSV:
                df.to_csv(output_path, index=False)
            elif format_type == DataFormat.EXCEL:
                df.to_excel(output_path, index=False)
            elif format_type == DataFormat.JSON:
                df.to_json(output_path, orient='records', indent=2)
            logger.info(f"Data successfully exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise

    def process_pipeline(self, input_path: Union[str, Path],
                         output_path: Union[str, Path],
                         output_format: Union[DataFormat, str] = DataFormat.CSV) -> Dict[str, Any]:
        """Complete data processing pipeline."""
        stats = {
            'original_rows': 0,
            'original_columns': 0,
            'rows_after_missing': 0,
            'outliers_detected': 0,
            'rows_after_outliers': 0,
            'status': 'failed',
            'error': None
        }

        try:
            # Load data
            df = self.load_data(input_path)
            stats['original_rows'] = len(df)
            stats['original_columns'] = len(df.columns)

            # Handle missing values
            df = self.handle_missing_values(df)
            stats['rows_after_missing'] = len(df)

            # Detect and handle outliers
            clean_df, outliers_df = self.detect_outliers(df)
            stats['outliers_detected'] = len(outliers_df)
            stats['rows_after_outliers'] = len(clean_df)

            # Standardize data
            clean_df = self.standardize_data(clean_df)

            # Export cleaned data
            self.export_data(clean_df, output_path, output_format)

            # Optionally export outliers
            if not outliers_df.empty:
                outlier_path = Path(output_path).with_stem(f"{Path(output_path).stem}_outliers")
                self.export_data(outliers_df, outlier_path, output_format)

            stats['status'] = 'success'
            return stats

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            stats['error'] = str(e)
            return stats


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


@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def index():
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

            config = {
                "outlier_contamination": float(request.form.get('contamination', 0.05)),
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

    if request.is_json:
        config = request.get_json()
    else:
        config = {
            "outlier_contamination": float(request.form.get('contamination', 0.05)),
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