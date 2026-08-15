#!/usr/bin/env bash
# Re-execute the published tutorials.
#
# This is the ONLY thing that should refresh notebook outputs. The docs build
# does not execute notebooks (see `execute: false` in mkdocs.yml), so outputs
# change when you decide they change -- which means every figure on the website
# corresponds to a run somebody actually looked at.
#
# docs/tutorials is a symlink to this directory, so there is nothing to copy
# afterwards: executing in place updates the site content.
#
# Needs the full scientific stack, plus manyTDE for tutorial 10.
# Takes roughly half an hour, most of it tutorial 10's chain.
#
# Uses `python3 -m nbconvert` rather than `jupyter nbconvert`: pip installs the
# jupyter console scripts into a bin/ directory that is often not on PATH (on a
# python.org macOS build they land in
# /Library/Frameworks/Python.framework/Versions/3.X/bin). The module form works
# regardless.
set -euo pipefail

PY="${PYTHON:-python3}"

if ! "$PY" -c "import nbconvert, fitted" >/dev/null 2>&1; then
    echo "Needs nbconvert and an importable FitTeD. Check with:"
    echo "    $PY -c 'import nbconvert, fitted'"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/tutorials"

# The published set. Keep in step with `nav` in mkdocs.yml.
NOTEBOOKS=(
    00_introduction_and_installation
    01_data_loading_basics
    02_observer_frame_data_input
    03_model_setup_and_parameters
    04_basic_fitting_workflow
    05_mcmc_fitting_with_emcee
    06_complete_workflow_at2019dsg
    07_model_options_reference
)

for nb in "${NOTEBOOKS[@]}"; do
    echo "=== $nb  $(date +%H:%M:%S)"
    "$PY" -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=7200 "$nb.ipynb"
done

cd "$REPO_ROOT"
"$PY" tools/check_notebook_outputs.py
echo
"$PY" tools/check_notebook_outputs.py
echo
echo "=== done. Review the diffs before committing:"
echo "      git diff --stat tutorials/"
echo "    Notebook diffs are mostly base64 image churn; check the printed"
echo "    numbers rather than trying to read the diff."
