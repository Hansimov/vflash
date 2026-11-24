# Develop and qualify a change

## Set up a checkout

Use Python 3.11 or newer. Install the editable project with `python -m pip install -e '.[dev,server]'`, then run `python -m pytest`. For a documentation change run `npm ci` and `npm run docs:build` from the repository root. Update both languages.

## Match the test to the contract

Use CPU cases for schemas, input rejection, ownership and cleanup. For a numerical or GPU change, record the actual device, profile, seed, source geometry, duration and audio settings. Decode the full result and compare motion, appearance and sound to the request. Keep reproducible public measurements and upstream attribution.

## Ship an upgrade

For a compatible user-facing feature publish a minor version; use a patch version for a compatible fix. Reserve prereleases for an explicitly limited evaluation. Build and install the wheel in a fresh environment, compare its version with the tag, then publish an ordinary GitHub Release and check the Latest link. Document migration steps for changed callers. Keep model and image inventories independently versioned.
