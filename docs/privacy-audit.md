# Privacy audit — October 10, 2026

This is a disclosure audit, not a guarantee that all content is suitable for
publication. No repository visibility or Git history was changed. The initial
snapshot was based on `f231e85be193f0dbfda0d978099860ac68ace37d`, plus the
uncommitted kernel and attribution work.

## Findings

| Area | Finding | Publication implication |
|---|---|---|
| Credential patterns | No live credential candidate found. Matches in `tests/test_prompt_studio.py` are explicit fake-key and rejected credential-URL fixtures. No matching credentials in commit messages. | No credential removal/rotation identified by this scan. This is pattern screening, not proof that an unknown secret format is absent. |
| Current machine paths | 53 files include user-home or mounted-volume paths, concentrated in benchmark JSON, logs and development/replay scripts. | Publishing them exposes machine usernames, volume/directory names and model/output locations. They remain in this candidate; paths were not silently edited in historical measurement records. |
| Historical machine paths | 46 distinct file paths have matching content in reachable Git history. | Deleting or redacting current files alone would not remove past disclosures. History was not rewritten. |
| Commit identity | Five distinct author/committer email addresses; three are not GitHub `noreply` addresses. | These addresses remain visible in historical commit metadata. They include contributor identities; no authorship was rewritten. |
| Repository media | 21 PNG, 43 JPEG and 28 MP4 files in the initial working snapshot. | Images/videos were checked for metadata, not visually reviewed for confidential content. Absence of sensitive metadata does not clear their depicted content. |
| Media metadata | No GPS EXIF, matching personal paths or credential candidates detected in the inspected metadata/container strings. PNG text/EXIF and JPEG headers were parsed without loading pixels; no metadata parse errors. | No metadata stripping was indicated by these checks. Video GPS fields and nonstandard/opaque metadata formats were not exhaustively decoded. |
| Local private state | No `.env`, credential/workspace JSON, PEM/key or SSH private-key filename candidate in the current snapshot or reachable historical filenames. `_runtime` and caches are ignored; Prompt Studio stores its workspace under ComfyUI's user directory. | Private runtime state is not included. Ignore rules now also protect common accidental credential/user/input/output/model additions. Ignored state was not read. |
| Embedded media | Current data-URI occurrences are implementation/test strings; none found in current JSON/workflow files. | No embedded reference upload was identified in those files. This does not replace visual review of separate media. |
| GitHub surfaces | Repository private; one branch (`main`), no tags, releases/release assets, Actions runs, issues, pull requests, discussions or forks at audit time. | No additional content exists on these surfaces to expose. Remote `main` matches the audited base revision. |

The [sanitized findings inventory](../benchmarks/data/privacy-audit-2026-10.json)
lists affected repository paths, counts and scope without reproducing machine
path values, email addresses, keys or prompts.

## Scope and method

- Inspected all blobs reachable through the local branch/remote refs: **1,080
  text blobs and 88 binary blobs**, plus the initial **857 current tracked or
  nonignored files**. The remote branch/tag inventory agrees with the local
  reference scope; it has no extra published branch or tag to audit. Rechecked
  the final 867-file candidate after restructuring: the disclosure findings and
  92 media-file count are unchanged.
- Screened private-key headers, common GitHub/OpenAI/AWS/Google/Slack credentials,
  JWTs, credential-bearing URLs and literal credential assignments. Reviewed
  matches in source tests to distinguish fixtures from live credentials.
- Inspected PNG text/EXIF chunks, JPEG header/EXIF data and printable MP4 container
  strings. No image pixels or video frames were decoded. Source checkpoint data
  was not interpreted as model weights.
- Counted author/committer addresses and checked commit messages. Inspected
  repository-relative names and machine-path patterns; raw values are omitted
  from the public report.
- Did not read anything on the SATA drive, personal clips, external workflow
  inputs, confidential Qwen input images/workflow, ignored runtime/user state,
  unreachable Git objects, reflogs or server logs. Remote issue/PR/discussion/fork
  counts were checked; all are zero.

## Remaining publication decisions

The release candidate is committed with these known disclosures documented.
The new commit uses the maintainer's GitHub `noreply` address; it does not change
historical identities.
Before making it public, decide whether to retain the benchmark path records and
historical email addresses, and verify that the benchmark media's visible content
is acceptable. Sanitizing current paths and historical paths are different tasks;
rewriting history requires a separate deliberate operation. This audit does not
authorize publication of confidential material or change repository visibility.
