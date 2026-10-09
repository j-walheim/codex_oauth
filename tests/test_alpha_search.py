"""Standalone Codex search is an opaque, credentialed proxy route."""

from email.message import Message
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

import codex_oauth_proxy as proxy


class SearchHandler(proxy.Handler):
    def __init__(self, body):
        encoded = json.dumps(body).encode()
        self.path = "/v1/alpha/search"
        self.command = "POST"
        self.headers = Message()
        self.headers["Content-Length"] = str(len(encoded))
        self.headers["Content-Type"] = "application/json"
        self.rfile = io.BytesIO(encoded)
        self.wfile = io.BytesIO()
        self.server = SimpleNamespace()
        self.status = None

    def send_response(self, code):
        self.status = code

    def send_header(self, name, value):
        pass

    def end_headers(self):
        pass


class AlphaSearchTests(unittest.TestCase):
    def test_route_preserves_request_and_structured_results(self):
        body = {"id": "search-test", "model": "gpt-6-sol",
                "commands": {"search_query": [{"q": "site:fda.gov endpoint guidance"}]}}
        result = {"output": "FDA result", "results": [{"type": "text_result",
                  "url": "https://www.fda.gov/example", "snippet": "Source passage"}]}
        handler = SearchHandler(body)
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=result)

        client_class = httpx.Client
        with patch.object(proxy, "resolve_creds", return_value=[
            proxy.Cred("fake-token", {}, "test", "test-id", "https://backend.test")
        ]), patch.object(proxy.httpx, "Client", side_effect=lambda **kwargs: client_class(
            transport=httpx.MockTransport(respond), **kwargs)):
            handler.route()

        self.assertEqual(handler.status, 200)
        self.assertEqual(str(requests[0].url), "https://backend.test/alpha/search")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer fake-token")
        self.assertEqual(json.loads(requests[0].content), body)
        self.assertEqual(json.loads(handler.wfile.getvalue()), result)


if __name__ == "__main__":
    unittest.main()
