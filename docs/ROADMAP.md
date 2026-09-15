# Feature Roadmap

Supervisor-provided issue list (`ISSUES TO SOLVE qml.pdf`, added 2026-09-15),
tracked against implementation status. The supervisor flagged six items as
"build these first" — marked **(priority)** below.

## Build these first

| Feature | Status | Notes |
|---|---|---|
| Bulk URL checker **(priority)** | **Done** | `/bulk` — paste up to 50 URLs or upload a `.csv`/`.txt` file, de-duplicated, each scanned with the Classical (Fair) SVM. `flask_app/templates/bulk.html`, `_parse_bulk_urls`/`_bulk_analyse_one` in `flask_app/app.py`. Live-verified on the deployed Render instance, including the "⚠ Unreachable" badge for sites that couldn't actually be fetched. |
| Downloadable analysis reports **(priority)** | **Partial** | Bulk results export to CSV (`POST /bulk-download`), including a `Reachable` Yes/No column. PDF/Excel export and single-URL report download are not yet built. |
| Shortened-URL expander **(priority)** | **Partial** | `flask_app/ssrf_guard.py::safe_get` already follows and validates every redirect hop (SSRF-safe, IP-pinned) instead of trusting `requests`' built-in redirect handling — the backend groundwork for this exists, but there's no dedicated "paste a short link, see the expanded destination" UI yet. |
| Lookalike domain detector **(priority)** | Not started | |
| Explainable URL risk report **(priority)** | Partial | The single-URL analyser already shows a feature-by-feature breakdown with suspicious flags and an estimated/live-fetched distinction; the bulk checker now flags per-row when a verdict is based on estimated rather than real fetched data (the "⚠ Unreachable" badge). A structured "which checks failed / why the result is uncertain" report is not yet built. |
| Scheduled URL monitoring **(priority)** | Not started | Needs persistent storage beyond the current ephemeral SQLite (see `docs/PROJECT_DOCUMENTATION.md` §5.1) plus a scheduler and notification channel. |

## Everything else from the issue list

URL watchlists, redirect-chain viewer (the data is already captured by
`check_site_health`'s `redirect_chain` field — no dedicated viewer UI yet),
lookalike/organization domain protection, QR-code URL checker, URL extraction
from documents, webpage link checker, sitemap URL scanner,
threat-intelligence cross-checking, URL-only fallback analysis (partially
covered by the existing "estimated feature" fallback), safe webpage preview,
historical URL timeline, "Learn This URL" mode, interactive URL learning lab,
bulk model benchmarking, browser extension, URL-checking API, team review
workspace, incorrect-verdict reporting — none started.

Source PDF kept at `related project documentation files/ISSUES TO SOLVE qml.pdf`
(outside this repo, on the author's local machine) — copy it into `docs/` if
it needs to travel with the repository.
