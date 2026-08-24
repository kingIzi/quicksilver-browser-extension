#!/usr/bin/env python
"""End-to-end test: run minzoto_ai.detect_mgm() against every Mux asset.

The AssetPipeline:
  1. lists every asset in the Mux environment (paginated),
  2. makes sure each asset exposes a downloadable `audio.m4a` static rendition,
     requesting one (and waiting for it) when missing,
  3. runs `minzoto_ai.detect_mgm(url)` on the rendition URL and prints the
     result for each asset,
  4. reports per-asset and whole-pipeline inference timings.

Usage:
    MUX_TOKEN_ID=... MUX_TOKEN_SECRET=... ./.venv/bin/python test_detect_mgm_mux.py
    ./.venv/bin/python test_detect_mgm_mux.py --limit 5      # smoke run
    ./.venv/bin/python test_detect_mgm_mux.py --no-prepare   # only assets that
                                                             # already have audio
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import mux_python
from mux_python.rest import ApiException

import minzoto_ai as m

# Credentials fall back to the working pair from main.py; prefer env vars.
MUX_TOKEN_ID = os.environ.get(
    "MUX_TOKEN_ID", "9f3a1b0b-185a-4535-b1e4-bf29bfcee181"
)
MUX_TOKEN_SECRET = os.environ.get(
    "MUX_TOKEN_SECRET",
    "wDV/omDFIGskEQd/5eJD+GRv+L4X65wo+QBzCuCJIdNCzI0v5iywR3vB24oCNqWHCaCGso+DzR2",
)

RENDITION_NAME = "audio.m4a"
STREAM_BASE = "https://stream.mux.com"


@dataclass
class AssetResult:
    """One asset's trip through the pipeline."""

    asset_id: str
    url: str | None = None
    prob: float | None = None
    verdict: str | None = None
    sample_rate: int | None = None
    inference_seconds: float | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PipelineReport:
    """Timings and outcomes for the whole AssetPipeline run."""

    results: list[AssetResult] = field(default_factory=list)
    prep_seconds: float = 0.0
    inference_wall_seconds: float = 0.0
    skipped_no_rendition: int = 0
    skipped_not_ready: int = 0

    @property
    def inference_times(self) -> list[float]:
        return [r.inference_seconds for r in self.results if r.inference_seconds is not None]

    @property
    def ok_results(self) -> list[AssetResult]:
        return [r for r in self.results if r.error is None]


def retry_after_seconds(exc: ApiException, fallback: float) -> float:
    """Seconds to wait after a 429: honor Retry-After when Mux sends it."""
    headers = getattr(exc, "headers", None) or {}
    value = None
    try:  # HTTPHeaderDict (urllib3) or plain dict
        value = headers.get("Retry-After")
    except AttributeError:
        pass
    if value is None:
        return fallback
    try:
        return max(float(value), 1.0)
    except ValueError:
        return fallback  # HTTP-date form; not worth parsing for a test


class AssetPipeline:
    """Lists Mux assets, resolves a playable audio URL for each, and runs
    minzoto_ai.detect_mgm() on it while tracking timing."""

    def __init__(self, token_id: str, token_secret: str):
        configuration = mux_python.Configuration()
        configuration.username = token_id
        configuration.password = token_secret
        self.assets_api = mux_python.AssetsApi(mux_python.ApiClient(configuration))

    # ------------------------------------------------------------------ Mux

    def list_assets(self) -> list:
        """Every asset in the environment, following cursor pagination."""
        assets: list = []
        cursor = None
        while True:
            kwargs = {"limit": 100}
            if cursor:
                kwargs["cursor"] = cursor
            resp = self.assets_api.list_assets(**kwargs)
            assets.extend(resp.data)
            cursor = getattr(resp, "next_cursor", None)
            if not cursor or not resp.data:
                return assets

    def has_audio_rendition(self, asset) -> bool:
        static = getattr(asset, "static_renditions", None)
        if not static:
            return False
        files = getattr(static, "files", None) or []
        return any(getattr(f, "name", None) == RENDITION_NAME for f in files)

    def request_audio_renditions(self, assets: list, pace: float = 0.5) -> int:
        """Kick off `audio-only` static rendition preparation for every asset
        missing one. Honors 429 Retry-After with exponential backoff. Returns
        the number of requests accepted."""
        accepted = 0
        for i, asset in enumerate(assets, 1):
            for attempt in range(6):
                try:
                    req = mux_python.CreateStaticRenditionRequest(resolution="audio-only")
                    self.assets_api.create_asset_static_rendition(asset.id, req)
                    accepted += 1
                    break
                except ApiException as e:
                    if e.status == 429:
                        wait = retry_after_seconds(e, fallback=5.0 * 2**attempt)
                        print(
                            f"    [{i}/{len(assets)}] {asset.id}: 429, "
                            f"backing off {wait:.0f}s (attempt {attempt + 1})",
                            flush=True,
                        )
                        time.sleep(wait)
                        continue
                    # e.g. already preparing; the poll phase waits either way.
                    print(f"    [{i}/{len(assets)}] {asset.id}: {e.status} {e.reason}", flush=True)
                    break
            else:
                print(f"    [{i}/{len(assets)}] {asset.id}: gave up after repeated 429s", flush=True)
            time.sleep(pace)  # pace creates well below the API rate limit
        return accepted

    def wait_for_renditions(self, assets: list, timeout: float, interval: float = 15.0) -> int:
        """Poll until every asset's audio rendition is ready (or timeout).
        Returns the number of assets ready when we stop waiting."""
        deadline = time.monotonic() + timeout
        pending = {a.id for a in assets}
        while pending and time.monotonic() < deadline:
            time.sleep(interval)
            for asset_id in list(pending):
                try:
                    asset = self.assets_api.get_asset(asset_id).data
                except ApiException:
                    time.sleep(1.0)  # likely rate limited; ease off briefly
                    continue
                if self.has_audio_rendition(asset):
                    pending.discard(asset_id)
                time.sleep(0.2)  # pace polling calls below the rate limit
            print(
                f"    renditions ready: {len(assets) - len(pending)}/{len(assets)} "
                f"(waiting on {len(pending)})",
                flush=True,
            )
        return len(assets) - len(pending)

    @staticmethod
    def audio_url(asset) -> str | None:
        """Public playback URL for the audio rendition, if one exists."""
        if not asset.playback_ids:
            return None
        playback_id = next(
            (p for p in asset.playback_ids if getattr(p, "policy", "public") == "public"),
            None,
        )
        if playback_id is None:
            return None
        return f"{STREAM_BASE}/{playback_id.id}/{RENDITION_NAME}"

    # -------------------------------------------------------------- inference

    def detect(self, asset) -> AssetResult:
        """Time a single detect_mgm() run against the asset's Mux URL."""
        result = AssetResult(asset_id=asset.id)
        result.url = self.audio_url(asset)
        if result.url is None:
            result.error = "no public playback id"
            return result

        started = time.perf_counter()
        try:
            detection = m.detect_mgm(result.url)
        except Exception as e:  # decode/network/model failure for this asset
            result.error = f"{type(e).__name__}: {e}"
            return result
        finally:
            result.inference_seconds = time.perf_counter() - started

        result.prob = detection["prob"]
        result.verdict = detection["verdict"]
        result.sample_rate = detection.get("sample_rate")
        return result


def format_seconds(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def run(args: argparse.Namespace) -> PipelineReport:
    pipeline = AssetPipeline(MUX_TOKEN_ID, MUX_TOKEN_SECRET)
    report = PipelineReport()

    print("Listing assets ...", flush=True)
    assets = pipeline.list_assets()
    ready = [a for a in assets if a.status == "ready" and a.playback_ids]
    print(
        f"{len(assets)} assets total, {len(ready)} ready with playback ids"
        f" (skipping {len(assets) - len(ready)})",
        flush=True,
    )

    # Phase 1 — make sure an audio.m4a rendition exists for every asset.
    prep_started = time.perf_counter()
    missing = [a for a in ready if not pipeline.has_audio_rendition(a)]
    if missing:
        if args.no_prepare:
            report.skipped_no_rendition = len(missing)
            print(
                f"--no-prepare: skipping {len(missing)} assets without renditions",
                flush=True,
            )
            ready = [a for a in ready if pipeline.has_audio_rendition(a)]
        else:
            print(
                f"Requesting audio-only renditions for {len(missing)} assets ...",
                flush=True,
            )
            pipeline.request_audio_renditions(missing, pace=args.pace)
            print("Waiting for renditions to finish encoding ...", flush=True)
            done = pipeline.wait_for_renditions(missing, args.prep_timeout)
            print(f"{done}/{len(missing)} renditions ready", flush=True)

            # Re-list once so URLs reflect the freshly prepared renditions.
            refreshed = []
            for a in ready:
                try:
                    refreshed.append(pipeline.assets_api.get_asset(a.id).data)
                except ApiException:
                    refreshed.append(a)
            ready = refreshed
            report.skipped_not_ready = sum(
                1 for a in ready if not pipeline.has_audio_rendition(a)
            )
            ready = [a for a in ready if pipeline.has_audio_rendition(a)]
    report.prep_seconds = time.perf_counter() - prep_started

    # Apply --limit after filtering so it counts only assets we can test.
    if args.limit:
        ready = ready[: args.limit]
        print(f"--limit {args.limit}: testing {len(ready)} assets", flush=True)

    # Phase 2 — inference over the whole pipeline, one URL at a time.
    print(f"\nRunning detect_mgm() on {len(ready)} Mux assets ...\n", flush=True)
    inference_started = time.perf_counter()
    for i, asset in enumerate(ready, 1):
        result = pipeline.detect(asset)
        report.results.append(result)
        if result.error:
            print(f"[{i}/{len(ready)}] {result.asset_id}: ERROR {result.error}", flush=True)
        else:
            print(
                f"[{i}/{len(ready)}] {result.asset_id}: "
                f"verdict={result.verdict} prob={result.prob:.4f} "
                f"({format_seconds(result.inference_seconds)})",
                flush=True,
            )
            print(f"    detect_mgm({result.url}) -> {result.as_dict()}", flush=True)
    report.inference_wall_seconds = time.perf_counter() - inference_started
    return report


def print_summary(report: PipelineReport, wall_seconds: float) -> None:
    times = report.inference_times
    verdicts: dict[str, int] = {}
    for r in report.ok_results:
        verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1

    print("\n" + "=" * 62)
    print("AssetPipeline summary")
    print("=" * 62)
    print(f"assets inferred      : {len(report.ok_results)}")
    print(f"errors               : {len(report.results) - len(report.ok_results)}")
    if report.skipped_no_rendition:
        print(f"skipped (no rend.)   : {report.skipped_no_rendition}")
    if report.skipped_not_ready:
        print(f"skipped (not ready)  : {report.skipped_not_ready}")
    if times:
        print(f"inference total      : {format_seconds(sum(times))}")
        print(f"inference wall clock : {format_seconds(report.inference_wall_seconds)}")
        print(
            "per asset            : "
            f"avg {format_seconds(sum(times) / len(times))}, "
            f"min {format_seconds(min(times))}, "
            f"max {format_seconds(max(times))}"
        )
    if verdicts:
        print(f"verdicts             : {verdicts}")
    print(f"rendition prep       : {format_seconds(report.prep_seconds)}")
    print(f"pipeline wall clock  : {format_seconds(wall_seconds)}")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="only test the first N assets (0 = all)")
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="do not request missing static renditions; skip those assets",
    )
    parser.add_argument(
        "--prep-timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for rendition encoding (default 1800)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.5,
        help="seconds between rendition-create calls (default 0.5)",
    )
    parser.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = parser.parse_args()

    started = time.perf_counter()
    report = run(args)
    wall_seconds = time.perf_counter() - started

    print_summary(report, wall_seconds)

    if args.json:
        payload = {
            "summary": {
                "assets_inferred": len(report.ok_results),
                "errors": len(report.results) - len(report.ok_results),
                "verdicts": {
                    r.verdict: sum(1 for x in report.ok_results if x.verdict == r.verdict)
                    for r in report.ok_results
                },
                "inference_total_seconds": sum(report.inference_times),
                "pipeline_wall_seconds": wall_seconds,
            },
            "results": [r.as_dict() for r in report.results],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.json}")
    return 0 if len(report.ok_results) == len(report.results) and report.results else 1


if __name__ == "__main__":
    sys.exit(main())
