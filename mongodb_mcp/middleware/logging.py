import time
import logging
from fastmcp.server.middleware import Middleware, MiddlewareContext

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    async def on_message(self, context: MiddlewareContext, call_next):
        logger.info(f"Request from {context.source}: {context.method} {context.message}")

        try:
            start_time = time.time()
            response = await call_next(context)
            process_time = (time.time() - start_time) * 1000
            logger.info(f"Response: {context.method} {context.message} - {response} ({process_time:.2f}ms)")

            return response
        except Exception as e:
            logger.exception(e)
