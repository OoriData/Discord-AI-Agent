# demo_server/sse_server_vendored.py
"""
SSE Server Transport Module

This module implements a Server-Sent Events (SSE) transport layer for MCP servers.

Example usage:
```
    # Create an SSE transport at an endpoint
    sse = SseServerTransport("/messages/")

    # Create Starlette routes for SSE and message handling
    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]

    # Define handler functions
    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )

    # Create and run Starlette app
    starlette_app = Starlette(routes=routes)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
```

See SseServerTransport class documentation for more details.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import ValidationError
from sse_starlette import EventSourceResponse
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

import mcp.types as types

logger = logging.getLogger(__name__)


class VendoredSseServerTransport:
    """
    SSE server transport for MCP. This class provides _two_ ASGI applications,
    suitable to be used with a framework like Starlette and a server like Hypercorn:

        1. connect_sse() is an ASGI application which receives incoming GET requests,
           and sets up a new SSE stream to send server messages to the client.
        2. handle_post_message() is an ASGI application which receives incoming POST
           requests, which should contain client messages that link to a
           previously-established SSE session.
    """

    _endpoint: str
    _read_stream_writers: dict[
        UUID, MemoryObjectSendStream[types.JSONRPCMessage | Exception]
    ]

    def __init__(self, endpoint: str) -> None:
        """
        Creates a new SSE server transport, which will direct the client to POST
        messages to the relative or absolute URL given.
        """

        super().__init__()
        self._endpoint = endpoint
        self._read_stream_writers = {}
        logger.info(f"SseServerTransport (FIXED/VENDORED version) initialized with endpoint: {endpoint}")

    @asynccontextmanager
    async def connect_sse(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            logger.error("connect_sse received non-HTTP request")
            raise ValueError("connect_sse can only handle HTTP requests")

        logger.debug("Setting up SSE connection")
        read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
        read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

        write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
        write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

        session_id = uuid4()
        session_uri = f"{quote(self._endpoint)}?session_id={session_id.hex}"
        self._read_stream_writers[session_id] = read_stream_writer
        logger.debug(f"Created new session with ID: {session_id}")

        sse_stream_writer, sse_stream_reader = anyio.create_memory_object_stream[
            dict[str, Any]
        ](0)

        async def sse_writer():
            logger.debug("Starting SSE writer")
            async with sse_stream_writer, write_stream_reader:
                await sse_stream_writer.send({"event": "endpoint", "data": session_uri})
                logger.debug(f"Sent endpoint event: {session_uri}")

                async for message in write_stream_reader:
                    logger.debug(f"Sending message via SSE: {message}")
                    await sse_stream_writer.send(
                        {
                            "event": "message",
                            "data": message.model_dump_json(
                                by_alias=True, exclude_none=True
                            ),
                        }
                    )

        async with anyio.create_task_group() as tg:
            response = EventSourceResponse(
                content=sse_stream_reader, data_sender_callable=sse_writer
            )
            logger.debug("Starting SSE response task")
            tg.start_soon(response, scope, receive, send)

            logger.debug("Yielding read and write streams")
            yield (read_stream, write_stream)

    async def handle_post_message(
            self, scope: Scope, receive: Receive, send: Send
        ) -> None:
            logger.debug("Handling POST message (FIXED/VENDORED Implementation)")
            request = Request(scope, receive)

            session_id_param = request.query_params.get("session_id")
            if session_id_param is None:
                logger.warning("Received request without session_id")
                # Use direct ASGI send for error response too for consistency, though Response() is fine here
                await send({
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(b"session_id is required")).encode())],
                })
                await send({"type": "http.response.body", "body": b"session_id is required", "more_body": False})
                return

            try:
                session_id = UUID(hex=session_id_param)
                logger.debug(f"Parsed session ID: {session_id}")
            except ValueError:
                logger.warning(f"Received invalid session ID: {session_id_param}")
                await send({
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(b"Invalid session ID")).encode())],
                })
                await send({"type": "http.response.body", "body": b"Invalid session ID", "more_body": False})
                return

            writer = self._read_stream_writers.get(session_id)
            if not writer:
                logger.warning(f"Could not find session for ID: {session_id}")
                await send({
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(b"Could not find session")).encode())],
                })
                await send({"type": "http.response.body", "body": b"Could not find session", "more_body": False})
                return

            body = await request.body()
            # Limit logging potentially large bodies
            logger.debug(f"Received POST body length: {len(body)}")
            if len(body) < 500: # Log small bodies only
                logger.debug(f"Received JSON: {body!r}")


            try:
                message = types.JSONRPCMessage.model_validate_json(body)
                logger.debug(f"Validated client message: {message.method if hasattr(message, 'method') else 'Notification/Response'}")
            except ValidationError as err:
                logger.error(f"Failed to parse message: {err}")
                # Send 400 Bad Request response
                err_body = b"Could not parse message"
                await send({
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(err_body)).encode())],
                })
                await send({"type": "http.response.body", "body": err_body, "more_body": False})
                # Optionally send the validation error internally if needed by other logic
                # await writer.send(err) # Uncomment if the MCPServer needs to know about validation errors
                return

            # --- Start of Corrected Response Handling ---
            logger.debug("Sending 202 Accepted status (no body/content-length)")
            # Manually send the 202 Accepted status without body/content-length
            await send(
                {
                    "type": "http.response.start",
                    "status": 202,
                    "headers": [
                        # NO content-length!
                        # Add minimal headers; Uvicorn/Starlette usually add date/server
                        (b"content-type", b"text/plain; charset=utf-8"), # Optional, signals no JSON body in *this* response
                    ],
                }
            )
            # Send an empty body to correctly terminate the *initial* HTTP response phase
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",  # Empty body
                    "more_body": False, # Signal end of this specific HTTP response
                }
            )
            logger.debug("Sent minimal 202 Accepted response components")
            # --- End of Corrected Response Handling ---

            # Pass the successfully parsed message to the internal MCP session handler
            logger.debug(f"Forwarding message to internal writer for session {session_id}")
            try:
                await writer.send(message)
                logger.debug(f"Successfully forwarded message to writer for session {session_id}")
            except Exception as e_send:
                # Log if sending to the internal stream fails (e.g., if the session was closed)
                logger.error(f"Failed to forward message to internal writer for session {session_id}: {e_send}")
                # Don't try to send another HTTP response here, connection might be gone
