"""OAuth2 client_credentials authentication for PISTE (Légifrance + Judilibre)."""

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

PISTE_TOKEN_URL = os.environ.get(
    "PISTE_TOKEN_URL",
    "https://oauth.piste.gouv.fr/api/oauth/token",
)


class PISTEAuth:
    """Manages OAuth2 tokens for the PISTE API platform."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self._client_id = client_id or os.environ["PISTE_CLIENT_ID"]
        self._client_secret = client_secret or os.environ["PISTE_CLIENT_SECRET"]
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _request_token(self) -> None:
        """Request a new OAuth2 token from PISTE."""
        client = await self._get_client()
        response = await client.post(
            PISTE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "openid",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._expires_at = time.time() + data.get("expires_in", 3600) - 30
        logger.info("PISTE token acquired, expires in %ds", data.get("expires_in", 3600))

    async def get_token(self) -> str:
        """Return a valid access token, refreshing if expired."""
        if self._token is None or time.time() >= self._expires_at:
            await self._request_token()
        return self._token  # type: ignore[return-value]

    async def get_auth_headers(self) -> dict[str, str]:
        """Return headers with a valid Bearer token."""
        token = await self.get_token()
        return {"Authorization": f"Bearer {token}"}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
