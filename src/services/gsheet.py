from datetime import datetime, timezone

import gspread


class GsheetService:
    def __init__(self) -> None:
        self.client = gspread.service_account()
    
    def append_row(self, table_name: str, data: list, date: datetime | None = None):
        # Open the spreadsheet by name
        sheet = self.client.open(table_name).sheet1

        # Data to post
        row = [date or datetime.now(timezone.utc), *data]
        sheet.append_row(row) # This adds the data to the next available row



gsheet_service = GsheetService()