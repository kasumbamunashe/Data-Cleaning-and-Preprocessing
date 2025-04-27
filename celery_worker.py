from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

celery = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)


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
            # Return just the filename for download
            return os.path.basename(output_path)
        else:
            raise Exception(result.get('error', 'Processing failed'))
    except Exception as e:
        # Clean up if needed
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)