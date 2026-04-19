from datetime import datetime, timezone

import gspread


class GsheetService:
    def __init__(self) -> None:
        self._client: gspread.Client | None = None

    @property
    def client(self) -> gspread.Client:
        if self._client is None:
            self._client = gspread.service_account()
        return self._client

    def append_row(
        self,
        table_name: str,
        data: list,
        date: datetime | None = None,
    ) -> None:
        sheet = self.client.open(table_name).sheet1
        timestamp = (date or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S")
        protected = [f"'{v}" if isinstance(v, str) else v for v in data]
        row = [timestamp, protected]
        sheet.append_row(row, value_input_option="USER_ENTERED")


gsheet_service = GsheetService()
