#!/usr/bin/env bash
# Bootstrap and launch jev-bench on a stock macOS or Ubuntu machine.
#
# Reuses whatever is already installed. It installs uv (the standalone build, into ~/.local/bin,
# shell profiles untouched) only when uv is missing, broken or older than MIN_UV. It installs a
# Python only when none satisfies pyproject's requires-python. Then it runs `jev-bench` from the
# locked environment. Safe to re-run.
#
# Usage: ./run.sh [jev-bench args...]      (no args = serve; e.g. ./run.sh serve --port 9000)
#
# Functions:
#   info, ok, warn, die  colored messages on stderr (NO_COLOR and non-TTY aware)
#   version_ge           dotted-version comparison
#   uv_usable            true if the given uv binary runs and is >= MIN_UV
#   find_uv              set UV to a usable uv from PATH or UV_DIR
#   as_root              run a command as root, directly or through sudo
#   ensure_downloader    make sure curl or wget exists (apt-get fallback on Ubuntu)
#   download             print a URL's body using curl or wget
#   install_uv           install the standalone uv into UV_DIR and set UV
#   ensure_python        make sure a Python matching requires-python is available
#   main                 cd to the repo, prepare uv and Python, exec jev-bench

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

readonly MIN_UV="0.9.5"
readonly UV_DIR="$HOME/.local/bin"
readonly UV_INSTALLER="https://astral.sh/uv/install.sh"

if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
  C_OK=$'\033[32m' C_WARN=$'\033[33m' C_ERR=$'\033[31m' C_DIM=$'\033[2m' C_OFF=$'\033[0m'
else
  C_OK='' C_WARN='' C_ERR='' C_DIM='' C_OFF=''
fi

info() { printf '%s%s%s\n' "$C_DIM" "$*" "$C_OFF" >&2; }
ok() { printf '%s%s%s\n' "$C_OK" "$*" "$C_OFF" >&2; }
warn() { printf '%swarning:%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die() { printf '%serror:%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

version_ge() {
  local have want i
  IFS=. read -r -a have <<<"$1"
  IFS=. read -r -a want <<<"$2"
  for i in 0 1 2; do
    [ "${have[i]:-0}" -gt "${want[i]:-0}" ] && return 0
    [ "${have[i]:-0}" -lt "${want[i]:-0}" ] && return 1
  done
  return 0
}

uv_usable() {
  local version
  { [ -n "$1" ] && [ -x "$1" ]; } || return 1
  version="$("$1" --version 2>/dev/null)" || return 1
  version="${version#uv }"
  version="${version%% *}"
  version_ge "${version%%[!0-9.]*}" "$MIN_UV"
}

find_uv() {
  local candidate
  for candidate in "$(command -v uv || true)" "$UV_DIR/uv"; do
    if uv_usable "$candidate"; then UV="$candidate" && return 0; fi
  done
  return 1
}

as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"
  elif command -v sudo >/dev/null 2>&1; then sudo "$@"
  else die "need root or sudo to run: $*"
  fi
}

ensure_downloader() {
  if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then return 0; fi
  command -v apt-get >/dev/null 2>&1 || die "install curl or wget, then re-run"
  info "Installing curl with apt-get"
  as_root apt-get update -qq
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates
}

download() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --proto '=https' --tlsv1.2 --retry 3 --connect-timeout 15 --max-time 120 "$1"
  else
    wget -qO- --timeout=30 --tries=3 "$1"
  fi
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    warn "$(command -v uv) is broken or older than uv $MIN_UV; installing a standalone uv"
  fi
  ensure_downloader
  info "Installing uv into $UV_DIR (shell profiles are not modified)"
  download "$UV_INSTALLER" | env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh ||
    die "uv installer failed"
  uv_usable "$UV_DIR/uv" || die "uv at $UV_DIR/uv does not run or is older than $MIN_UV"
  UV="$UV_DIR/uv"
}

ensure_python() {
  local request
  request="$(sed -n "s/^requires-python *= *[\"']\([^\"']*\)[\"'].*/\1/p" pyproject.toml)"
  [ -n "$request" ] || die "requires-python not found in pyproject.toml"
  if "$UV" python find "$request" >/dev/null 2>&1; then return 0; fi
  info "Installing Python $request with uv"
  "$UV" python install --no-bin "$request"
}

main() {
  cd -- "$(dirname -- "$0")"
  find_uv || install_uv
  info "Using $("$UV" --version) at $UV"
  ensure_python
  [ "$#" -gt 0 ] || set -- serve
  ok "Starting: jev-bench $*"
  exec "$UV" run --locked --no-dev jev-bench "$@"
}

main "$@"
