#!/usr/bin/env bash
#
# Environment setup for the yolov8 fault-injection project.
#
#   source setup.sh              set up both toolchains in the current shell
#   source setup.sh --check      report what was found, change nothing
#   ./setup.sh --install         create the venv and install Python deps
#   source setup.sh --python     Python only
#   source setup.sh --rtl        RTL/C++ toolchain only
#
# Part 1 is Python (model, fault-injection campaigns, vector generation).
# Part 2 is the RTL and C++ toolchain (Verilator, VCS, Yosys).
# The two are independent: you can do RTL work without Python and vice versa,
# except that `make vectors` regenerates test vectors from the Python model.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${YOLOV8_VENV:-$ROOT/.venv}"

MODE_PYTHON=1
MODE_RTL=1
DO_INSTALL=0
CHECK_ONLY=0

for a in "$@"; do
    case "$a" in
        --install) DO_INSTALL=1 ;;
        --check)   CHECK_ONLY=1 ;;
        --python)  MODE_RTL=0 ;;
        --rtl)     MODE_PYTHON=0 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; return 0 2>/dev/null || exit 0 ;;
        *) echo "setup.sh: unknown option $a" >&2; return 1 2>/dev/null || exit 1 ;;
    esac
done

ok()   { printf '  \e[0;32m%-22s\e[0m %s\n' "$1" "$2"; }
warn() { printf '  \e[0;33m%-22s\e[0m %s\n' "$1" "$2"; }
bad()  { printf '  \e[0;31m%-22s\e[0m %s\n' "$1" "$2"; }

# ===========================================================================
# Part 1 -- Python
# ===========================================================================
setup_python() {
    echo "[1] Python"

    # System python3.11 is 3.11.9 on ece-linux. Do not use plain `python3`:
    # that is 3.6.8, which no current torch supports.
    local PY=""
    for c in python3.11 python3.12; do
        command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
    done
    if [ -z "$PY" ]; then
        bad "python3.11" "not found -- see GETTING_STARTED.md"
        return 1
    fi

    if [ "$DO_INSTALL" = 1 ] && [ "$CHECK_ONLY" = 0 ]; then
        if [ ! -d "$VENV" ]; then
            echo "      creating venv at $VENV with $($PY --version)"
            "$PY" -m venv "$VENV" || return 1
        fi
        "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
        echo "      installing requirements.txt (torch pulls ~2.5 GB of CUDA wheels)"
        # /tmp is a 4 GB partition on these hosts and pip unpacks every wheel
        # there, so the CUDA wheels overflow it as "No space left on device"
        # while /home has terabytes free. Keep the cache off AFS too: the home
        # volume has a 100 GB quota that a few torch installs will eat.
        local scratch="${TMPDIR_LOCAL:-/home/$USER/.cache}"
        mkdir -p "$scratch/pip" "$scratch/tmp" 2>/dev/null
        if [ -d "$scratch/tmp" ]; then
            export TMPDIR="$scratch/tmp"
            export PIP_CACHE_DIR="$scratch/pip"
            echo "      TMPDIR=$TMPDIR  (/tmp is too small for the CUDA wheels)"
        fi

        # pypi.org must be the PRIMARY index. Left to itself pip resolves the
        # 150-200 MB nvidia-* wheels via pypi.nvidia.com, which truncates them
        # (nvidia-nccl-cu12 stalls at 162/189 MB every time); PyPI serves the
        # identical wheels without trouble.
        local try
        for try in 1 2 3; do
            "$VENV/bin/python" -m pip install \
                --index-url https://pypi.org/simple \
                --extra-index-url https://download.pytorch.org/whl/cu121 \
                --retries 10 --resume-retries 10 \
                -r "$ROOT/requirements.txt" && break
            [ "$try" = 3 ] && { bad "pip install" "failed after 3 attempts"; return 1; }
            warn "pip install" "attempt $try failed, retrying"
        done
    fi

    if [ -x "$VENV/bin/python" ]; then
        [ "$CHECK_ONLY" = 0 ] && { . "$VENV/bin/activate"; }
        ok "venv" "$VENV ($("$VENV/bin/python" --version 2>&1))"
    else
        warn "venv" "absent -- run ./setup.sh --install"
    fi

    # src/ holds weightSentry, fi_lib and shield, which the scripts import by
    # bare name rather than as a package.
    if [ "$CHECK_ONLY" = 0 ]; then
        case ":$PYTHONPATH:" in
            *":$ROOT/src:"*) ;;
            *) export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
    fi
    ok "PYTHONPATH" "$ROOT/src"

    if [ -x "$VENV/bin/python" ]; then
        local v
        v=$("$VENV/bin/python" - <<'EOF' 2>/dev/null
try:
    import torch, ultralytics
    print(f"torch {torch.__version__}  ultralytics {ultralytics.__version__}  cuda={torch.cuda.is_available()}")
except Exception as e:
    print("IMPORT FAILED:", e)
EOF
)
        case "$v" in
            IMPORT*) bad "packages" "$v" ;;
            *)       ok  "packages" "$v" ;;
        esac
    fi

    # Checkpoints are gitignored (639 MB, three over GitHub's 100 MB limit).
    # Ultralytics fetches them on first use; yolov8n.pt is the one the RTL
    # vectors are generated from, so it is needed for `make vectors`.
    if [ -f "$ROOT/yolov8n.pt" ]; then
        ok "yolov8n.pt" "present"
    else
        warn "yolov8n.pt" "absent -- auto-downloaded on first model load"
    fi
}

# ===========================================================================
# Part 2 -- RTL and C++ toolchain
# ===========================================================================
setup_rtl() {
    echo "[2] RTL / C++ toolchain"

    # VCS: the ece240 script picks the version matching the OS (RHEL 8 ->
    # T-2022.06, RHEL 9 -> Y-2026.03) and sets LM_LICENSE_FILE. Do not hardcode
    # a version; this host family runs both releases.
    local VCS_SETUP=/afs/ece.cmu.edu/class/ece240/bin/setup_vcs
    if command -v vcs >/dev/null 2>&1; then
        ok "vcs" "$(command -v vcs)"
    elif [ -r "$VCS_SETUP" ]; then
        if [ "$CHECK_ONLY" = 0 ]; then
            . "$VCS_SETUP" >/dev/null 2>&1
            command -v vcs >/dev/null 2>&1 \
                && ok "vcs" "$(command -v vcs)" \
                || bad "vcs" "setup_vcs ran but vcs is still not on PATH"
        else
            warn "vcs" "available via $VCS_SETUP"
        fi
    else
        bad "vcs" "not found and $VCS_SETUP unreadable (AFS token? run: ki)"
    fi

    if command -v verilator >/dev/null 2>&1; then
        ok "verilator" "$(verilator --version 2>&1 | head -1)"
    else
        bad "verilator" "not found -- needed by 'make test'"
    fi

    # Yosys is not packaged on these hosts; it comes from a VTR build. Point
    # YOSYS at your own build if you have one.
    local YOSYS_FALLBACK=/afs/ece.cmu.edu/usr/mgong2/tools/vtr-verilog-to-routing/build/bin
    if command -v yosys >/dev/null 2>&1; then
        ok "yosys" "$(command -v yosys)"
    elif [ -x "$YOSYS_FALLBACK/yosys" ]; then
        [ "$CHECK_ONLY" = 0 ] && export PATH="$YOSYS_FALLBACK:$PATH"
        ok "yosys" "$YOSYS_FALLBACK/yosys"
    else
        warn "yosys" "not found -- only 'make synth' needs it"
    fi

    command -v g++ >/dev/null 2>&1 \
        && ok "g++" "$(g++ -dumpversion)" \
        || bad "g++" "not found -- Verilator compiles its output with it"

    # SAED32 EDK: standard cells, IO pads and SRAM macros, for gate-level
    # simulation and power once there is a netlist.
    local SAED=/afs/ece.cmu.edu/support/synopsys/synopsys.release/SAED32_EDK-01312018
    [ -d "$SAED" ] && { [ "$CHECK_ONLY" = 0 ] && export SAED32_EDK="$SAED"; ok "SAED32_EDK" "$SAED"; } \
                   || warn "SAED32_EDK" "not found (only needed for gate-level sim)"
}

[ "$MODE_PYTHON" = 1 ] && setup_python
[ "$MODE_PYTHON" = 1 ] && [ "$MODE_RTL" = 1 ] && echo
[ "$MODE_RTL" = 1 ] && setup_rtl

if [ "$CHECK_ONLY" = 0 ] && [ "$DO_INSTALL" = 0 ]; then
    echo
    echo "ready.  python campaign:  python src/campaign.py --help"
    echo "        rtl tests:        cd hw && make test && make test-read"
fi
