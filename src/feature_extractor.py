import re
import socket
import urllib.parse

PHISHING_KEYWORDS = [
    'login', 'signin', 'verify', 'account', 'secure', 'banking',
    'update', 'confirm', 'password', 'credential', 'paypal', 'ebay',
    'amazon', 'microsoft', 'apple', 'google', 'facebook',
]

SHORTENERS = {
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
    'is.gd', 'buff.ly', 'adf.ly', 'rb.gy', 'short.to',
}

SUSPICIOUS_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.gq', '.pw',
    '.top', '.click', '.loan', '.work',
}

BRANDS = [
    'paypal', 'google', 'apple', 'microsoft', 'amazon', 'facebook',
    'netflix', 'ebay', 'instagram', 'twitter', 'linkedin', 'chase',
    'wellsfargo', 'bankofamerica', 'citibank',
]

COMMON_TLDS = ['com', 'net', 'org', 'edu', 'gov', 'io', 'co', 'uk', 'de', 'fr']

# Column order must match the training dataset exactly
FEATURE_COLUMNS = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens',
    'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore',
    'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon',
    'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www',
    'nb_com', 'nb_dslash', 'http_in_path', 'https_token',
    'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port',
    'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain',
    'nb_subdomains', 'prefix_suffix', 'random_domain',
    'shortening_service', 'path_extension', 'nb_redirection',
    'nb_external_redirection', 'length_words_raw', 'char_repeat',
    'shortest_words_raw', 'shortest_word_host', 'shortest_word_path',
    'longest_words_raw', 'longest_word_host', 'longest_word_path',
    'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints',
    'domain_in_brand', 'brand_in_subdomain', 'brand_in_path',
    'suspecious_tld', 'statistical_report', 'nb_hyperlinks',
    'ratio_intHyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks',
    'nb_extCSS', 'ratio_intRedirection', 'ratio_extRedirection',
    'ratio_intErrors', 'ratio_extErrors', 'login_form',
    'external_favicon', 'links_in_tags', 'submit_email',
    'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe',
    'popup_window', 'safe_anchor', 'onmouseover', 'right_clic',
    'empty_title', 'domain_in_title', 'domain_with_copyright',
    'whois_registered_domain', 'domain_registration_length',
    'domain_age', 'web_traffic', 'dns_record', 'google_index',
    'page_rank',
]

# Features computed directly from URL string (shown as "extracted" in UI)
URL_COMPUTED = {
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens',
    'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore',
    'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon',
    'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www',
    'nb_com', 'nb_dslash', 'http_in_path', 'https_token',
    'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port',
    'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain',
    'nb_subdomains', 'prefix_suffix', 'random_domain',
    'shortening_service', 'path_extension', 'length_words_raw',
    'char_repeat', 'shortest_words_raw', 'shortest_word_host',
    'shortest_word_path', 'longest_words_raw', 'longest_word_host',
    'longest_word_path', 'avg_words_raw', 'avg_word_host',
    'avg_word_path', 'phish_hints', 'domain_in_brand',
    'brand_in_subdomain', 'brand_in_path', 'suspecious_tld',
    'dns_record',
}


def _tokenize(text):
    words = re.split(r'[^a-zA-Z0-9]', text)
    return [w for w in words if w]


def _word_stats(words):
    if not words:
        return 0, 0, 0.0
    lengths = [len(w) for w in words]
    return min(lengths), max(lengths), round(sum(lengths) / len(lengths), 4)


def extract_features(url: str) -> tuple[dict, set]:
    """
    Returns (features_dict, estimated_set) where estimated_set contains
    feature names that could not be computed from the URL alone.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    parsed    = urllib.parse.urlparse(url)
    hostname  = parsed.hostname or ''
    path      = parsed.path or ''
    url_lower = url.lower()
    parts     = hostname.split('.')

    domain_part = parts[-2] if len(parts) >= 2 else hostname
    subdomain   = '.'.join(parts[:-2]) if len(parts) > 2 else ''

    url_words  = _tokenize(url)
    host_words = _tokenize(hostname)
    path_words = _tokenize(path)

    # ── URL character counts ──────────────────────────────────────────────────
    nb_space = url.count(' ') + url.count('%20')

    # ── Digit ratios ─────────────────────────────────────────────────────────
    ratio_digits_url  = round(sum(c.isdigit() for c in url) / max(len(url), 1), 4)
    ratio_digits_host = round(sum(c.isdigit() for c in hostname) / max(len(hostname), 1), 4)

    # ── IP used as hostname ───────────────────────────────────────────────────
    try:
        socket.inet_aton(hostname)
        ip = 1
    except (socket.error, OSError):
        ip = 0

    # ── Subdomain & domain structure ─────────────────────────────────────────
    abnormal_subdomain = 1 if re.search(r'[0-9]+-[a-z]|[a-z]+-[0-9]+', subdomain) else 0
    nb_subdomains      = max(0, len(parts) - 2) if len(parts) > 2 else 0

    alpha = [c for c in domain_part.lower() if c.isalpha()]
    vowels = set('aeiouAEIOU')
    consonant_ratio = sum(1 for c in alpha if c not in vowels) / max(len(alpha), 1)
    random_domain   = 1 if (len(domain_part) > 12 and consonant_ratio > 0.65) else 0

    # ── TLD features ─────────────────────────────────────────────────────────
    tld = '.' + parts[-1].lower() if parts else ''
    tld_in_path      = 1 if any(f'.{t}' in path.lower() for t in COMMON_TLDS) else 0
    tld_in_subdomain = 1 if any(t in subdomain.lower() for t in COMMON_TLDS) else 0

    # ── Word statistics ───────────────────────────────────────────────────────
    char_repeat_matches = re.findall(r'(.)\1+', url)
    char_repeat = max((len(m) + 1 for m in char_repeat_matches), default=0)

    s_raw,  l_raw,  avg_raw  = _word_stats(url_words)
    s_host, l_host, avg_host = _word_stats(host_words)
    s_path, l_path, avg_path = _word_stats(path_words)

    # ── Phishing hint & brand checks ─────────────────────────────────────────
    phish_hints        = sum(1 for kw in PHISHING_KEYWORDS if kw in url_lower)
    domain_in_brand    = 1 if any(b in domain_part.lower() for b in BRANDS) else 0
    brand_in_subdomain = 1 if any(b in subdomain.lower() for b in BRANDS) else 0
    brand_in_path      = 1 if any(b in path.lower() for b in BRANDS) else 0

    # ── DNS record check ─────────────────────────────────────────────────────
    try:
        socket.gethostbyname(hostname)
        dns_record = 1
    except socket.error:
        dns_record = 0

    # ── Assemble all features in dataset column order ─────────────────────────
    f = {
        'length_url':               len(url),
        'length_hostname':          len(hostname),
        'ip':                       ip,
        'nb_dots':                  url.count('.'),
        'nb_hyphens':               url.count('-'),
        'nb_at':                    url.count('@'),
        'nb_qm':                    url.count('?'),
        'nb_and':                   url.count('&'),
        'nb_or':                    url.count('|'),
        'nb_eq':                    url.count('='),
        'nb_underscore':            url.count('_'),
        'nb_tilde':                 url.count('~'),
        'nb_percent':               url.count('%'),
        'nb_slash':                 url.count('/'),
        'nb_star':                  url.count('*'),
        'nb_colon':                 url.count(':'),
        'nb_comma':                 url.count(','),
        'nb_semicolumn':            url.count(';'),
        'nb_dollar':                url.count('$'),
        'nb_space':                 nb_space,
        'nb_www':                   url_lower.count('www'),
        'nb_com':                   url_lower.count('.com'),
        'nb_dslash':                url.count('//'),
        'http_in_path':             1 if 'http' in path.lower() else 0,
        'https_token':              1 if parsed.scheme == 'https' else 0,
        'ratio_digits_url':         ratio_digits_url,
        'ratio_digits_host':        ratio_digits_host,
        'punycode':                 1 if 'xn--' in hostname.lower() else 0,
        'port':                     1 if (parsed.port and parsed.port not in (80, 443)) else 0,
        'tld_in_path':              tld_in_path,
        'tld_in_subdomain':         tld_in_subdomain,
        'abnormal_subdomain':       abnormal_subdomain,
        'nb_subdomains':            nb_subdomains,
        'prefix_suffix':            1 if '-' in domain_part else 0,
        'random_domain':            random_domain,
        'shortening_service':       1 if any(s in hostname.lower() for s in SHORTENERS) else 0,
        'path_extension':           1 if re.search(r'\.(php|html|htm|asp|aspx|cgi|pl)(\?|$)', path.lower()) else 0,
        # ── defaults for features needing page content / external APIs ────────
        'nb_redirection':           0,
        'nb_external_redirection':  0,
        'length_words_raw':         len(url_words),
        'char_repeat':              char_repeat,
        'shortest_words_raw':       s_raw,
        'shortest_word_host':       s_host,
        'shortest_word_path':       s_path,
        'longest_words_raw':        l_raw,
        'longest_word_host':        l_host,
        'longest_word_path':        l_path,
        'avg_words_raw':            avg_raw,
        'avg_word_host':            avg_host,
        'avg_word_path':            avg_path,
        'phish_hints':              phish_hints,
        'domain_in_brand':          domain_in_brand,
        'brand_in_subdomain':       brand_in_subdomain,
        'brand_in_path':            brand_in_path,
        'suspecious_tld':           1 if tld in SUSPICIOUS_TLDS else 0,
        'statistical_report':       0,
        'nb_hyperlinks':            0,
        'ratio_intHyperlinks':      0.0,
        'ratio_extHyperlinks':      0.0,
        'ratio_nullHyperlinks':     0.0,
        'nb_extCSS':                0,
        'ratio_intRedirection':     0.0,
        'ratio_extRedirection':     0.0,
        'ratio_intErrors':          0.0,
        'ratio_extErrors':          0.0,
        'login_form':               0,
        'external_favicon':         0,
        'links_in_tags':            0.0,
        'submit_email':             0,
        'ratio_intMedia':           0.0,
        'ratio_extMedia':           0.0,
        'sfh':                      0,
        'iframe':                   0,
        'popup_window':             0,
        'safe_anchor':              0.0,
        'onmouseover':              0,
        'right_clic':               0,
        'empty_title':              0,
        'domain_in_title':          0,
        'domain_with_copyright':    0,
        'whois_registered_domain':  0,
        'domain_registration_length': -1,
        'domain_age':               -1,
        'web_traffic':              0,
        'dns_record':               dns_record,
        'google_index':             0,
        'page_rank':                0,
    }

    estimated = set(FEATURE_COLUMNS) - URL_COMPUTED
    return f, estimated
