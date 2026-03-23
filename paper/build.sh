#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
PDF_PATH="$BUILD_DIR/main.pdf"

mkdir -p "$BUILD_DIR"

if [ -f "$PDF_PATH" ]; then
  printf "Override %s? [y/N] " "$PDF_PATH"
  read -r ANSWER
  case "$ANSWER" in
    y|Y) ;;
    *) exit 0 ;;
  esac
fi

cd "$ROOT_DIR"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
printf "Built %s\n" "$PDF_PATH"
