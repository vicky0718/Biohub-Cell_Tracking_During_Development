"""A minimal Kaggle client for autonomous operation, built on urllib.

**Why not the official CLI.** Two independent blockers in this environment:

* `kaggle>=1.7` talks to ``api.kaggle.com``, which this container's egress proxy denies
  outright (``connect_rejected ... gateway answered 403 to CONNECT``). Only
  ``www.kaggle.com`` is permitted.
* `kaggle==1.6.17` does target ``www.kaggle.com/api/v1``, but its swagger-generated
  client builds its own SSL context and ignores ``REQUESTS_CA_BUNDLE`` /
  ``SSL_CERT_FILE``, so it cannot be pointed at the proxy's CA bundle.

`urllib.request` uses the system trust store, which already carries the proxy CA, so it
works untouched. TLS verification is never disabled anywhere in this module.

**Credentials** are read from ``~/.kaggle/kaggle.json`` at call time and never returned,
logged, or embedded in anything this module writes. Nothing here prints a request header.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import tempfile
import time
import zipfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = "https://www.kaggle.com/api/v1"
CONFIG = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"


class KaggleError(RuntimeError):
    """A non-2xx response. Carries the status and body so failures are diagnosable."""

    def __init__(self, status, body, path):
        self.status, self.body, self.path = status, body, path
        super().__init__(f"{path} -> HTTP {status}: {body[:500]}")


def _auth_header() -> str:
    c = json.loads(CONFIG.read_text())
    raw = f"{c['username']}:{c['key']}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def username() -> str:
    return json.loads(CONFIG.read_text())["username"]


def _request(method: str, path: str, *, data: bytes | None = None,
             content_type: str | None = None, timeout: int = 300,
             absolute: bool = False, extra_headers: dict | None = None):
    url = path if absolute else f"{BASE}{path}"
    headers = {"Authorization": _auth_header(), "User-Agent": "claude-biohub/1.0"}
    if content_type:
        headers["Content-Type"] = content_type
    headers.update(extra_headers or {})
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        raise KaggleError(e.code, e.read().decode("utf-8", "replace"), path) from None


def get_json(path: str, **params):
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    _, body = _request("GET", f"{path}?{q}" if q else path)
    return json.loads(body or b"null")


def post_json(path: str, payload: dict, timeout: int = 300):
    _, body = _request("POST", path, data=json.dumps(payload).encode(),
                       content_type="application/json", timeout=timeout)
    return json.loads(body or b"null")


# --------------------------------------------------------------------- kernels

def kernel_push(slug: str, notebook_path: str | Path, *, title: str,
                enable_gpu: bool = False, enable_internet: bool = True,
                dataset_sources: list[str] | None = None,
                competition_sources: list[str] | None = None,
                kernel_sources: list[str] | None = None,
                machine_shape: str | None = None,
                is_private: bool = True) -> dict:
    """Create or update a kernel and start a run. Returns the push response.

    ``is_private`` defaults to True and every call site should leave it that way: a
    public notebook hands the work to every other competitor.
    """
    nb = json.loads(Path(notebook_path).read_text())
    # `id` is the NUMERIC kernel id and must be omitted when creating or updating by
    # name; the owner/name pair goes in `slug`, and the display name in `newTitle`.
    # Sending the slug as `id` fails with "Could not convert string to integer".
    payload = {
        "slug": f"{username()}/{slug}",
        "newTitle": title,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": is_private,
        "enableGpu": enable_gpu,
        "enableInternet": enable_internet,
        "datasetDataSources": dataset_sources or [],
        "competitionDataSources": competition_sources or [],
        "kernelDataSources": kernel_sources or [],
        "modelDataSources": [],
        "categoryIds": [],
        "text": json.dumps(nb),
    }
    # Kaggle's plain `enableGpu` hands out a Tesla P100 (sm_60), and the image's
    # torch 2.10+cu128 ships kernels only for sm_70+ -- CUDA reports "available" and
    # then every launch dies with "no kernel image is available for execution on the
    # device". T4 is sm_75, and has tensor cores the P100 lacks, so fp16 autocast is
    # faster there too.
    if machine_shape:
        payload["machineShape"] = machine_shape
    return post_json("/kernels/push", payload)


def kernel_status(slug: str, user: str | None = None) -> dict:
    return get_json("/kernels/status", userName=user or username(), kernelSlug=slug)


def kernel_output(slug: str, user: str | None = None) -> dict:
    return get_json("/kernels/output", userName=user or username(), kernelSlug=slug)


def kernel_wait(slug: str, *, poll: int = 60, timeout: int = 13 * 3600,
                on_tick=None) -> dict:
    """Block until the kernel leaves a running state. Returns the final status dict.

    ``poll`` is deliberately coarse. The run is minutes to hours; polling faster only
    burns requests.
    """
    t0 = time.time()
    while True:
        st = kernel_status(slug)
        status = (st.get("status") or "").lower()
        if on_tick:
            on_tick(status, time.time() - t0, st)
        if status not in ("running", "queued", "queueing", ""):
            return st
        if time.time() - t0 > timeout:
            return {**st, "status": "TIMEOUT_WAITING", "waited_s": time.time() - t0}
        time.sleep(poll)


# -------------------------------------------------------------------- datasets

def dataset_status(owner: str, slug: str) -> dict:
    return get_json(f"/datasets/status/{owner}/{slug}")


def dataset_list(user: str | None = None) -> list:
    return get_json("/datasets/list", user=user or username())


def _upload_one(path: Path, name: str | None = None) -> dict:
    """Two-step blob upload: reserve a slot, then PUT the bytes to the signed URL.

    **This endpoint cannot create directories.** ``name`` is passed through, but Kaggle
    strips any path from it and the file lands at the dataset root either way — measured,
    not assumed: v51 uploaded with `path.name` and v52 with `p.relative_to(folder)`, and a
    probe kernel listing the mount showed both as 19 flat files with no `harness/` or
    `pipeline/`. `claude_budget2` died 61 s in on ``our repo  None`` against both.

    A directory tree therefore has to go up as a **zip**, which Kaggle extracts on ingest.
    `dataset_new_version` does that automatically; see its docstring.
    """
    size = path.stat().st_size
    last_mod = int(path.stat().st_mtime)
    # The MODERN blob endpoint, not the legacy /datasets/upload/file/{size}/{mtime}.
    # The legacy route does return a createUrl and a token, but the token carries no
    # path, and /datasets/create/version then rejects it with "Path must be non-null".
    res = post_json("/blobs/upload", {
        "type": "dataset",
        "name": name or path.name,
        "contentLength": size,
        "contentType": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "lastModifiedEpochSeconds": last_mod,
    })
    create_url = res.get("createUrl") or res.get("CreateUrl")
    token = res.get("token") or res.get("Token")
    if not create_url:
        raise KaggleError(0, json.dumps(res), "datasets/upload/file")
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = path.read_bytes()
    # The signed URL is pre-authorised; sending our Basic header there would leak the
    # credential to a third-party host, so build this request WITHOUT it.
    req = urllib.request.Request(create_url, data=body, method="PUT",
                                 headers={"Content-Type": ctype,
                                          "Content-Length": str(len(body))})
    with urllib.request.urlopen(req, timeout=1800) as f:
        if f.status not in (200, 201):
            raise KaggleError(f.status, f.read().decode("utf-8", "replace"), create_url)
    return {"token": token}


def dataset_version_number(owner: str, slug: str) -> int | None:
    """The dataset's current version number, or None if it cannot be read."""
    try:
        return int(get_json(f"/datasets/view/{owner}/{slug}").get("currentVersionNumber"))
    except Exception:
        return None


def dataset_new_version(owner: str, slug: str, folder: Path | str, notes: str,
                        *, delete_old_versions: bool = False, wait: bool = True,
                        poll: int = 10, timeout: int = 1800) -> dict:
    """Upload every file in ``folder`` as a NEW VERSION of an existing dataset.

    ``delete_old_versions`` is forced False: the standing rule is new versions only,
    never destroy what is already there.

    ``wait`` blocks until the version number ACTUALLY INCREMENTS. Do not substitute
    `dataset_status`: it reports "ready" for the version already published, so straight
    after a push it returns ready immediately and a kernel launched on that signal
    attaches the PREVIOUS version. That cost a run -- claude-detector-refine died 302 s in
    on a guard for a Config field that was in the upload but not yet in the attached
    version. The guard turned a silent wrong answer into a loud fast failure; this makes
    the failure not happen.

    **A folder containing subdirectories is uploaded as a single zip**, because the blob
    endpoint cannot create directories (see `_upload_one`) and Kaggle extracts zips on
    ingest. Uploading such a tree file-by-file flattens it, which silently breaks every
    consumer that looks for a package directory: it published `biohub-cell-tracking` v51
    and v52 with `harness/` and `pipeline/` gone, and `find_dir` in four notebooks looks
    for exactly those. v53 went up as a zip and a probe kernel confirmed the tree. A flat
    folder is still uploaded file-by-file, so existing single-directory callers are
    unchanged.
    """
    folder = Path(folder)
    before = dataset_version_number(owner, slug)
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    has_tree = any(p.parent != folder for p in files)
    if has_tree:
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / f"{slug}.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in files:
                    zf.write(p, p.relative_to(folder).as_posix())
            tokens = [_upload_one(zpath)]
    else:
        tokens = [_upload_one(p) for p in files]
    res = post_json(
        f"/datasets/create/version/{owner}/{slug}",
        {"versionNotes": notes, "files": tokens, "subtitle": None, "description": None,
         "isPrivate": True, "convertToCsv": False,
         "categoryIds": [], "deleteOldVersions": False},
        timeout=1800,
    )
    if not wait:
        return res
    t0 = time.time()
    while time.time() - t0 < timeout:
        now = dataset_version_number(owner, slug)
        ready = str(dataset_status(owner, slug)).lower().strip('"') == "ready"
        if now is not None and (before is None or now > before) and ready:
            res["version_number"] = now
            res["waited_s"] = round(time.time() - t0, 1)
            return res
        time.sleep(poll)
    raise KaggleError(0, f"version did not advance past {before} within {timeout}s "
                         f"(now {dataset_version_number(owner, slug)})",
                      f"datasets/create/version/{owner}/{slug}")


# ---------------------------------------------------------------- competitions

def submissions_list(competition: str) -> list:
    return get_json(f"/competitions/submissions/list/{competition}")


def leaderboard(competition: str) -> dict:
    return get_json(f"/competitions/{competition}/leaderboard/view")


__all__ = ["KaggleError", "username", "get_json", "post_json", "kernel_push",
           "dataset_version_number",
           "kernel_status", "kernel_output", "kernel_wait", "dataset_status",
           "dataset_list", "dataset_new_version", "submissions_list", "leaderboard"]


def kernel_push_like(slug: str, notebook_path: str | Path, *, title: str,
                     reference: str, require: tuple[str, ...] = (),
                     is_private: bool = True, **overrides) -> dict:
    """Push a kernel reusing ANOTHER kernel's whole source configuration.

    Retyping a source list is how `claude_submit_ratio` v1 died: an inspection printed only
    ``datasetDataSources``, the ``kernelDataSources`` entry carrying the torch wheelhouse
    never made it into the push, and the run drew a P100 that the image torch cannot run.
    The notebook's own markdown listed the wheelhouse as a required input; the push did not.

    ``reference`` is the slug of a kernel known to have worked. Every source list, plus the
    GPU and internet flags, is copied from it. ``require`` names substrings that MUST appear
    somewhere in the resulting sources -- a loud failure here beats an 8-second CUDA death.
    """
    ref = get_json("/kernels/pull", userName=username(),
                   kernelSlug=reference).get("metadata", {})
    cfg = {
        "dataset_sources": ref.get("datasetDataSources") or [],
        "competition_sources": ref.get("competitionDataSources") or [],
        "kernel_sources": ref.get("kernelDataSources") or [],
        "enable_gpu": bool(ref.get("enableGpu")),
        "enable_internet": bool(ref.get("enableInternet")),
    }
    cfg.update(overrides)
    everything = " ".join(cfg["dataset_sources"] + cfg["competition_sources"]
                          + cfg["kernel_sources"])
    missing = [r for r in require if r not in everything]
    if missing:
        raise KaggleError(0, f"reference {reference} does not supply {missing}; "
                             f"sources were {everything}", "kernels/push")
    return kernel_push(slug, notebook_path, title=title, is_private=is_private, **cfg)
