"""
SSRF guard for the live URL analyser.

Every route in app.py fetches, resolves, or connects to a URL that an
end user typed in — with the app now publicly deployed, that means any
visitor can ask the server to make requests on their behalf. Without
validation this is textbook SSRF: a submitted URL could point at
127.0.0.1, an internal 10.x/192.168.x address, or (critically, on cloud
hosts) the 169.254.169.254 metadata endpoint.

The fix classifies the *resolved IP*, never the hostname string, so
numeric/hex-encoded loopback forms (http://2130706433/, http://0x7f.0.0.1/)
can't slip through. Every redirect hop is re-validated and the connection
is pinned to the exact IP that was checked (never re-resolved at connect
time), which closes both the DNS-rebinding and redirect-to-internal
attack paths.
"""

import ipaddress
import socket
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.connection import create_connection as _urllib3_create_connection

ALLOWED_SCHEMES = {'http', 'https'}
ALLOWED_PORTS = {80, 443}


class BlockedURLError(Exception):
    """Raised when a URL/IP is not a safe public target."""


def _ip_is_public(ip_str):
    ip = ipaddress.ip_address(ip_str)
    mapped = getattr(ip, 'ipv4_mapped', None)
    if mapped:
        ip = mapped
    if ip.is_private or ip.is_loopback or ip.is_link_local or \
       ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return False
    return True


def resolve_and_validate(hostname, port):
    """Resolves hostname, requires every returned address to be public, returns one pinned IP."""
    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise BlockedURLError(f'DNS resolution failed for {hostname}: {e}') from e

    ips = {info[4][0] for info in infos}
    if not ips:
        raise BlockedURLError(f'no addresses resolved for {hostname}')

    for ip in ips:
        if not _ip_is_public(ip):
            raise BlockedURLError(f'{hostname} resolves to non-public address {ip}')

    return sorted(ips)[0]


def parse_and_check_url(url, allowed_ports=ALLOWED_PORTS):
    """Returns (scheme, hostname, port, pinned_ip). Raises BlockedURLError if unsafe."""
    full = url if url.startswith(('http://', 'https://')) else 'http://' + url
    parsed = urlparse(full)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise BlockedURLError(f'scheme {parsed.scheme!r} not allowed')

    hostname = parsed.hostname
    if not hostname:
        raise BlockedURLError('URL has no hostname')

    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    if allowed_ports is not None and port not in allowed_ports:
        raise BlockedURLError(f'port {port} not allowed')

    pinned_ip = resolve_and_validate(hostname, port)
    return parsed.scheme, hostname, port, pinned_ip


class _PinnedIPHTTPAdapter(HTTPAdapter):
    """Forces the connection to dial a pre-validated IP while keeping the
    original hostname for the Host header, TLS SNI, and certificate checks."""

    def __init__(self, pinned_ip, *args, **kwargs):
        self._pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        conn = super().get_connection_with_tls_context(request, verify, proxies=proxies, cert=cert)
        pinned_ip = self._pinned_ip

        class _PinnedHTTPSConnection(conn.ConnectionCls):
            def _new_conn(self):
                return _urllib3_create_connection(
                    (pinned_ip, self.port), self.timeout, source_address=self.source_address
                )

        conn.ConnectionCls = _PinnedHTTPSConnection
        return conn

    def get_connection(self, url, proxies=None):
        # Older urllib3/requests fallback path (no TLS-context hook available).
        conn = super().get_connection(url, proxies=proxies)
        pinned_ip = self._pinned_ip

        class _PinnedHTTPConnection(conn.ConnectionCls):
            def _new_conn(self):
                return _urllib3_create_connection(
                    (pinned_ip, self.port), self.timeout, source_address=self.source_address
                )

        conn.ConnectionCls = _PinnedHTTPConnection
        return conn


def safe_get(url, *, timeout=8, headers=None, max_redirects=5):
    """
    SSRF-safe replacement for requests.get(url, allow_redirects=True).
    Validates and pins every redirect hop instead of trusting requests'
    built-in redirect handling (which re-resolves DNS with no seam to
    re-check, and would happily follow a redirect to an internal address).
    """
    current = url
    visited = set()
    history = []

    for _ in range(max_redirects + 1):
        if current in visited:
            raise BlockedURLError('redirect loop detected')
        visited.add(current)

        _, hostname, port, pinned_ip = parse_and_check_url(current)

        session = requests.Session()
        adapter = _PinnedIPHTTPAdapter(pinned_ip)
        session.mount('https://', adapter)
        session.mount('http://', adapter)

        resp = session.get(current, timeout=timeout, headers=headers, allow_redirects=False)

        if resp.is_redirect and resp.headers.get('Location'):
            history.append(resp)
            current = urljoin(current, resp.headers['Location'])
            continue

        resp.history = history
        return resp

    raise BlockedURLError('too many redirects')
