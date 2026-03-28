import asyncio

import pytest

from skyward.daemon.protocol import Ping, Pong, SubmitTask, TaskSucceeded
from skyward.daemon.wire import encode, decode, async_send, async_recv


class TestEncoding:
    def test_roundtrip_simple(self) -> None:
        msg = Ping()
        data = encode(msg)
        assert isinstance(data, bytes)
        assert len(data) > 8  # 8-byte header + payload
        result = decode(data)
        assert isinstance(result, Ping)

    def test_roundtrip_with_payload(self) -> None:
        msg = SubmitTask(pool_name="train", payload=b"hello", timeout=60.0, client_id="c1")
        result = decode(encode(msg))
        assert isinstance(result, SubmitTask)
        assert result.pool_name == "train"
        assert result.payload == b"hello"
        assert result.client_id == "c1"

    def test_roundtrip_large_payload(self) -> None:
        """Verify payloads larger than 4GB header would allow."""
        large = b"x" * (2**20)  # 1MB
        msg = TaskSucceeded(payload=large)
        result = decode(encode(msg))
        assert isinstance(result, TaskSucceeded)
        assert len(result.payload) == 2**20


class TestAsyncTransport:
    @pytest.mark.asyncio
    async def test_send_recv_over_socket_pair(self) -> None:
        received: list = []

        async def server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            msg = await async_recv(reader)
            received.append(msg)
            await async_send(writer, Pong())
            writer.close()

        sock_path = "/tmp/test-skyward-wire.sock"
        srv = await asyncio.start_unix_server(server, path=sock_path)

        reader, writer = await asyncio.open_unix_connection(sock_path)
        await async_send(writer, Ping())
        resp = await async_recv(reader)
        writer.close()
        srv.close()

        assert isinstance(received[0], Ping)
        assert isinstance(resp, Pong)
