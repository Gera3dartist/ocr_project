"""Tests for the Google Sheets publishing service."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.services.gsheet import GsheetService


def _make_service_with_mock_client() -> tuple[GsheetService, MagicMock]:
    service = GsheetService()
    mock_client = MagicMock()
    service._client = mock_client
    return service, mock_client


def test_append_row_formats_datetime_for_sheets_date_picker() -> None:
    """Timestamp must be 'YYYY-MM-DD HH:MM:SS' so Google Sheets parses it as a date.

    Regression: ISO 8601 (with 'T', microseconds, and '+00:00') is stored as a
    plain string in Sheets, which breaks filtering, sorting, and the date picker.
    """
    service, mock_client = _make_service_with_mock_client()
    fixed_time = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)

    service.append_row("sheet_name", data=["0", "3", "8", "1", "4"], date=fixed_time)

    sheet = mock_client.open.return_value.sheet1
    sent_row = sheet.append_row.call_args[0][0]
    assert sent_row[0] == "2026-04-19 12:00:00"


def test_append_row_uses_user_entered_value_input() -> None:
    """Sheets only parses the date string if value_input_option='USER_ENTERED'.

    The default 'RAW' stores the literal text without type inference, so the
    date picker never activates.
    """
    service, mock_client = _make_service_with_mock_client()

    service.append_row("sheet_name", data=["0", "3"])

    sheet = mock_client.open.return_value.sheet1
    _, kwargs = sheet.append_row.call_args
    assert kwargs.get("value_input_option") == "USER_ENTERED"


def test_append_row_default_timestamp_matches_sheets_format() -> None:
    """Default timestamp must use the same Sheets-friendly format."""
    service, mock_client = _make_service_with_mock_client()

    service.append_row("sheet_name", data=["0", "3"])

    sheet = mock_client.open.return_value.sheet1
    sent_row = sheet.append_row.call_args[0][0]
    assert isinstance(sent_row[0], str)
    assert "T" not in sent_row[0]
    assert "+" not in sent_row[0]
    datetime.strptime(sent_row[0], "%Y-%m-%d %H:%M:%S")


def test_append_row_forwards_data_cells_after_timestamp() -> None:
    """Data cells must follow the timestamp in order."""
    service, mock_client = _make_service_with_mock_client()
    fixed_time = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)

    service.append_row("sheet_name", data=["0", "3", "8", "1", "4"], date=fixed_time)

    sheet = mock_client.open.return_value.sheet1
    sent_row = sheet.append_row.call_args[0][0]
    assert sent_row[1:] == ["0", "3", "8", "1", "4"]
