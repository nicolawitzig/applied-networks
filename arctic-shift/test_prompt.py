#!/usr/bin/env python3
"""Test harness: run the two-stage pipeline on known failure cases and compare to expected."""

from pathlib import Path

from classify_content import build_text_sample, classify_user, load_user_data

MODEL = "qwen3:14b"
MAX_CHARS = 1200
INPUT_DIRS = [Path("results/user_posts_jan_jun"), Path("results/user_posts_jul_dec")]

# (username, expected_label, description_of_prior_bug)
CASES = [
    # accepted-to-ETH previously misclassified as external
    ("lost_leopard_",        "decentral_individual/student",  "accepted-to-ETH → external"),
    ("AdEastern6906",        "decentral_individual/student",  "accepted-to-ETH → external"),
    ("tinatsh",              "decentral_individual/student",  "accepted-to-ETH → external"),
    ("Feisty_Anywhere4422",  "decentral_individual/student",  "accepted-to-ETH → external"),
    # currently-at-ETH previously misclassified as external
    ("PaintCharacter8338",   "decentral_individual/student",  "at-ETH → external"),
    ("BeginningSelection57", "decentral_individual/student",  "at-ETH → external"),
    # no-affiliation previously misclassified as external or decentral
    ("Background-Fish-8465", "unknown",                       "no-affiliation → external"),
    ("Vegetable-Brother-31", "unknown",                       "friend-mention → decentral"),
    # label-evidence contradiction: TU Munich evidence but decentral_individual label
    ("4g4o",                 "external_individual/student",   "TU-Munich evidence → decentral label"),
    # second-person text treated as self-identification
    ("04whizkid",            "unknown",                       "second-person Cambridge → external"),
    # preference statement treated as affiliation
    ("0attention-span",      "unknown",                       "viellicht-ZHAW preference → external"),
    # joke-format steps treated as applicant evidence
    ("0x-Error",             "unknown",                       "joke-steps → applicant"),
    # ZHAW alumni classified as current student instead of other
    ("0b00000110",           "external_individual/other",     "ZHAW-grad-in-industry → external/student"),
]


def find_user_file(username: str) -> Path:
    """Search INPUT_DIRS for user JSON; assert it exists in at least one."""
    matches = [d / f"{username}.json" for d in INPUT_DIRS if (d / f"{username}.json").exists()]
    assert matches, f"No JSON file found for {username!r} in {INPUT_DIRS}"
    return matches[0]


def main() -> None:
    sep = "─" * 72
    passed = failed = 0
    for username, expected, bug_desc in CASES:
        path = find_user_file(username)
        user = load_user_data(path)
        text = build_text_sample(user, MAX_CHARS)
        n_posts = len(user.posts) + len(user.comments)

        clf = classify_user(username, text, MODEL, n_posts, debug=False)
        actual = f"{clf.voice_type}/{clf.subtype}" if clf.subtype else clf.voice_type

        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        print(sep)
        print(f"[{status}] {username}  ({bug_desc})")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}  (conf={clf.confidence:.2f})")
        print(f"  evidence: {clf.evidence[:100]}")
        print(f"  reason:   {clf.reasoning[:100]}")

    print(sep)
    print(f"\nResults: {passed}/{passed + failed} passed")


if __name__ == "__main__":
    main()
