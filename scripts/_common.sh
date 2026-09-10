#!/usr/bin/env bash
# Shared settings for the reproduction scripts. Source this; do not run it.
set -euo pipefail

VIDORE_V1="vidore/arxivqa_test_subsampled vidore/docvqa_test_subsampled vidore/infovqa_test_subsampled \
vidore/tabfquad_test_subsampled vidore/tatdqa_test vidore/shiftproject_test \
vidore/syntheticDocQA_artificial_intelligence_test vidore/syntheticDocQA_energy_test \
vidore/syntheticDocQA_government_reports_test vidore/syntheticDocQA_healthcare_industry_test"

CACHE=${CACHE:-./cache}
OUT=${OUT:-./out}
CODES=${CODES:-"2 4 8 16 32 64"}

# The standard protocol: fit once on the public ColPali training collection, apply frozen, score
# every query of every evaluation subset.
STD="--train_on vidore/colpali_train_set --train_on_split train --eval_all_queries --refiner zero_init"

mkdir -p "$OUT/logs"
