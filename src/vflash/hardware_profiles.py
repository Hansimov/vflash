"""Explicit heterogeneous CUDA profiles; measured support is documented per GPU and mode."""

REMOTE_ARCHITECTURES = {
    "sm90": ("9.0", 75, "resident"),
    "sm103": ("10.3", 31, "block-ring"),
    "sm120": ("12.0", 30, "block-ring"),
}
V01_PROFILES = {
    f"{mode}-turbo4-v01-544-exact-{arch}"
    for mode in ("i2va", "fl2va")
    for arch in ("sm89", *REMOTE_ARCHITECTURES)
}


def base_profile(identifier):
    if identifier in V01_PROFILES:
        return identifier.rsplit("-", 1)[0] + "-sm89"
    return identifier


def target_id(architecture):
    return "remote-" + architecture + "-bf16-preview"
