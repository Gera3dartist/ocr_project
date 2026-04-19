"""Tests for the Google Sheets publishing service."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.services.gsheet import GsheetService


def _make_service_with_mock_client() -> tuple[GsheetService, MagicMock]:
    service = GsheetService()
    mock_client = MagicMock()
    service._client = mock_client
    return service, mock_client


def test_append_row_serializes_datetime_to_iso_string() -> None:
    """Timestamp must be a string so gspread can JSON-serialize the request body.

    Regression: passing a raw datetime crashed the edge device with
    "Object of type datetime is not JSON serializable" inside gspread's HTTP layer.
    """
    service, mock_client = _make_service_with_mock_client()
    fixed_time = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)

    service.append_row("sheet_name", data=["0", "3", "8", "1", "4"], date=fixed_time)

    sheet = mock_client.open.return_value.sheet1
    sent_row = sheet.append_row.call_args[0][0]
    assert sent_row[0] == fixed_time.isoformat()
    assert not any(isinstance(cell, datetime) for cell in sent_row)


def test_append_row_default_timestamp_is_iso_string() -> None:
    """When no date is provided, the default timestamp must also be stringified."""
    service, mock_client = _make_service_with_mock_client()

    service.append_row("sheet_name", data=["0", "3"])

    sheet = mock_client.open.return_value.sheet1
    sent_row = sheet.append_row.call_args[0][0]
    assert isinstance(sent_row[0], str)
    assert "T" in sent_row[0]


def test_append_row_forwards_data_cells_after_timestamp() -> None:
    """Data cells must follow the timestamp in order."""
    service, mock_client = _make_service_with_mock_client()
    fixed_time = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)

    service.append_row("sheet_name", data=["0", "3", "8", "1", "4"], date=fixed_time)

    sheet = mock_client.open.return_value.sheet1
    sent_row = sheet.append_row.call_args[0][0]
    assert sent_row[1:] == ["0", "3", "8", "1", "4"]
