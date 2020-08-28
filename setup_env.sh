#!/usr/bin/env bash
readonly REQUIREMENTS_DIRECTORY="/tapas_service"
readonly SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

function main() {
  echo "Install Tapas Service environment."
  echo "Install dependencies from ${REQUIREMENTS_DIRECTORY}"

  set -o errexit
  set -o pipefail
  set -o nounset
  set -o errtrace

  install_env

}

function install_env() {

  echo "[***] Printing Python Version..."
  python -V

  echo "[***] Run the Tapas Service Application..."

  python app.py

}

main
