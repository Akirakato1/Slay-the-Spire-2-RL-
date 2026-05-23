from sts2_rl.live.sts2mcp_client import Sts2McpClient


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(("GET", url, timeout))
        if url.endswith("/"):
            return FakeResponse(
                {"status": "ok", "message": "Hello from STS2 MCP v0.4.0"}
            )
        return FakeResponse({"state_type": "map"})

    def post(self, url, json, timeout):
        self.calls.append(("POST", url, json, timeout))
        return FakeResponse({"status": "ok", "message": "done"})


def test_health_calls_root_endpoint():
    fake = FakeSession()
    client = Sts2McpClient(base_url="http://localhost:15526", session=fake)

    assert client.health()["status"] == "ok"
    assert fake.calls[0][1] == "http://localhost:15526/"


def test_get_singleplayer_state_uses_json_format():
    fake = FakeSession()
    client = Sts2McpClient(base_url="http://localhost:15526", session=fake)

    assert client.get_singleplayer_state()["state_type"] == "map"
    assert fake.calls[0][1] == "http://localhost:15526/api/v1/singleplayer?format=json"


def test_post_singleplayer_action_posts_body():
    fake = FakeSession()
    client = Sts2McpClient(base_url="http://localhost:15526", session=fake)

    result = client.post_singleplayer_action({"action": "end_turn"})

    assert result["status"] == "ok"
    assert fake.calls[0][2] == {"action": "end_turn"}
