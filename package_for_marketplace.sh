#!/usr/bin/env bash
# Package SmartFit Orientation for UltiMaker Marketplace.
# Marketplace expects: smartfit_orientation/plugin.json (folder name = package_id)

set -e
cd "$(dirname "$0")"
OUT="smartfit_orientation"
ZIP="smartfit_orientation.zip"

rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"
mkdir -p "$OUT/qml_qt5" "$OUT/qml_qt6"

# Core plugin files
cp plugin.json __init__.py SmartFitOrientation.py CalculateOrientationJob.py MeshTweaker.py "$OUT/"
cp LICENSE NOTICE README.md CHANGELOG.md CMakeLists.txt "$OUT/"
cp -r SmartFitOrientation "$OUT/"
cp qml_qt5/SettingsPopup.qml "$OUT/qml_qt5/"
cp qml_qt6/SettingsPopup.qml "$OUT/qml_qt6/"

zip -r "$ZIP" "$OUT"
rm -rf "$OUT"

echo "Created $ZIP (root folder: $OUT/)"
echo "Upload this zip to the UltiMaker Marketplace."
