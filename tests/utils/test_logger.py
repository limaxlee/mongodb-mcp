import pytest
import logging
import zipfile
from io import BytesIO

from mongodb_mcp.utils.logger import LogConfig, initialize_logger, get_logs_zip_file, _create_logs_zip_file


class TestInitializeLogger:
    def test_creates_log_dir_when_missing(self, mocker, tmp_path):
        log_dir = tmp_path / "logs"
        mocker.patch.object(LogConfig, "LOG_DIR", str(log_dir))

        logger = logging.getLogger("test_creates_log_dir")
        logger.handlers.clear()

        initialize_logger("test.log", logger=logger)

        assert log_dir.is_dir()
        assert (log_dir / "test.log").exists()

    def test_uses_existing_log_dir(self, mocker, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        mocker.patch.object(LogConfig, "LOG_DIR", str(log_dir))

        logger = logging.getLogger("test_uses_existing_log_dir")
        logger.handlers.clear()

        result = initialize_logger("existing.log", logger=logger)

        assert result is logger
        assert (log_dir / "existing.log").exists()

    def test_attaches_stream_and_file_handlers(self, mocker, tmp_path):
        mocker.patch.object(LogConfig, "LOG_DIR", str(tmp_path))

        logger = logging.getLogger("test_attaches_handlers")
        logger.handlers.clear()

        initialize_logger("handlers.log", logger=logger)

        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types
        assert logging.handlers.RotatingFileHandler in handler_types
        assert logger.level == LogConfig.LEVEL

    def test_uses_root_logger_when_none_provided(self, mocker, tmp_path):
        mocker.patch.object(LogConfig, "LOG_DIR", str(tmp_path))

        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        root_logger.handlers.clear()

        try:
            result = initialize_logger("root.log")
            assert result is root_logger
            assert len(root_logger.handlers) >= 2
        finally:
            root_logger.handlers.clear()
            for h in original_handlers:
                root_logger.addHandler(h)


class TestGetLogsZipFile:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_logs(self, mocker, tmp_path):
        mocker.patch.object(LogConfig, "LOG_DIR", str(tmp_path))

        result = await get_logs_zip_file()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_zip_bytes_when_logs_exist(self, mocker, tmp_path):
        log_file = tmp_path / "mongodb_mcp_server.log"
        log_file.write_text("sample log content")
        mocker.patch.object(LogConfig, "LOG_DIR", str(tmp_path))

        result = await get_logs_zip_file()

        assert isinstance(result, bytes)
        with zipfile.ZipFile(BytesIO(result)) as z:
            names = z.namelist()
            assert "mongodb_mcp_server.log" in names
            assert z.read("mongodb_mcp_server.log") == b"sample log content"

    @pytest.mark.asyncio
    async def test_runs_in_executor(self, mocker, tmp_path):
        log_file = tmp_path / "executor.log"
        log_file.write_text("executor log content")
        mocker.patch.object(LogConfig, "LOG_DIR", str(tmp_path))

        mock_loop = mocker.MagicMock()

        async def fake_run_in_executor(executor, func):
            return func()

        mock_loop.run_in_executor = mocker.AsyncMock(side_effect=fake_run_in_executor)
        mocker.patch("mongodb_mcp.utils.logger.asyncio.get_running_loop", return_value=mock_loop)

        result = await get_logs_zip_file()

        assert isinstance(result, bytes)
        mock_loop.run_in_executor.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_propagates_executor_exception(self, mocker, tmp_path):
        log_file = tmp_path / "error.log"
        log_file.write_text("error log content")
        mocker.patch.object(LogConfig, "LOG_DIR", str(tmp_path))

        mock_loop = mocker.MagicMock()
        mock_loop.run_in_executor = mocker.AsyncMock(side_effect=RuntimeError("Executor error"))
        mocker.patch("mongodb_mcp.utils.logger.asyncio.get_running_loop", return_value=mock_loop)

        with pytest.raises(RuntimeError, match="Executor error"):
            await get_logs_zip_file()


class TestCreateLogsZipFile:
    def test_creates_zip_with_single_file(self, tmp_path):
        log_file = tmp_path / "single.log"
        log_file.write_text("single log content")

        result = _create_logs_zip_file([str(log_file)])

        assert isinstance(result, bytes)
        with zipfile.ZipFile(BytesIO(result)) as z:
            assert z.namelist() == ["single.log"]
            assert z.read("single.log") == b"single log content"

    def test_creates_zip_with_multiple_files(self, tmp_path):
        log_a = tmp_path / "a.log"
        log_a.write_text("content a")
        log_b = tmp_path / "b.log"
        log_b.write_text("content b")

        result = _create_logs_zip_file([str(log_a), str(log_b)])

        with zipfile.ZipFile(BytesIO(result)) as z:
            names = set(z.namelist())
            assert names == {"a.log", "b.log"}
            assert z.read("a.log") == b"content a"
            assert z.read("b.log") == b"content b"

    def test_raises_on_missing_file(self, tmp_path):
        missing = tmp_path / "missing.log"

        with pytest.raises(FileNotFoundError):
            _create_logs_zip_file([str(missing)])
