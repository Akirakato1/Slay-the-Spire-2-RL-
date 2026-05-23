from __future__ import annotations

from typing import Any, Protocol

import requests


class HttpSession(Protocol):
    def get(self, url: str, timeout: float):
        raise NotImplementedError

    def post(self, url: str, json: dict[str, Any], timeout: float):
        raise NotImplementedError


class Sts2McpClient:
    def __init__(
        self,
        base_url: str = "http://localhost:15526",
        timeout_s: float = 2.0,
        session: HttpSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def get_singleplayer_state(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/v1/singleplayer?format=json",
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def post_singleplayer_action(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/v1/singleplayer",
            json=body,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()
