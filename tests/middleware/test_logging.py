import pytest

from mongodb_mcp.middleware import LoggingMiddleware


class TestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_logs_and_calls_next(self, mocker, caplog):
        middleware = LoggingMiddleware()
        ctx = mocker.MagicMock()
        ctx.type = "request"
        ctx.source = "client"
        ctx.method = "tools/list"
        ctx.message = "payload"
        call_next = mocker.AsyncMock(return_value={"ok": True})

        with caplog.at_level("INFO"):
            result = await middleware.on_message(ctx, call_next)

        call_next.assert_awaited_once_with(ctx)
        assert result == {"ok": True}

        assert any("tools/list" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_handle_exceptions(self, mocker, caplog):
        middleware = LoggingMiddleware()
        ctx = mocker.MagicMock()
        ctx.type = "request"
        ctx.source = "client"
        ctx.method = "x"
        ctx.message = "y"
        call_next = mocker.AsyncMock(side_effect=RuntimeError("Connector error"))

        with caplog.at_level("ERROR"):
            result = await middleware.on_message(ctx, call_next)

        assert result is None
        call_next.assert_awaited_once_with(ctx)
