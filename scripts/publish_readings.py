

from pathlib import Path

from src.config import load_config
from src.services.measurements import make_readings
from src.services.gsheet import gsheet_service

PROJECT_ROOT = Path(__file__).parent.parent



def publish_data():
    config = load_config(str(PROJECT_ROOT / "config.json"))
    readings = make_readings()
    gsheet_service.append_row(
        table_name=config.gsheet_file_name,
        data=[readings.digits]
    )

    

if __name__ == '__main__':
    publish_data()