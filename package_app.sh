#!/usr/bin/env bash
# Build "PenPlot Studio.app" and put it on the Desktop.
#
# The bundle is a launcher, not a copy: it runs the project in place, so the
# app always starts the current code and stays a few hundred kilobytes instead
# of the ~400 MB a frozen Qt build would take. The trade-off is that the
# project folder has to stay where it is - the launcher says so if it moves.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
APP="${1:-$HOME/Desktop/PenPlot Studio.app}"
NAME="PenPlot Studio"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# ---- icon -----------------------------------------------------------------
ICONSET="$PROJECT/build/AppIcon.iconset"
if [ -f "$PROJECT/build/icon-1024.png" ]; then
  rm -rf "$ICONSET"; mkdir -p "$ICONSET"
  for size in 16 32 128 256 512; do
    sips -z $size $size "$PROJECT/build/icon-1024.png" \
         --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z $double $double "$PROJECT/build/icon-1024.png" \
         --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
  rm -rf "$ICONSET"
fi

# ---- launcher -------------------------------------------------------------
cat > "$APP/Contents/MacOS/PenPlotStudio" <<LAUNCHER
#!/bin/bash
PROJECT="$PROJECT"
if [ ! -x "\$PROJECT/.venv/bin/python" ] && [ ! -f "\$PROJECT/run.sh" ]; then
  osascript -e 'display alert "PenPlot Studio" message "The project folder has moved.\n\nExpected it at:\n$PROJECT\n\nRun package_app.sh again from its new location." as critical'
  exit 1
fi
cd "\$PROJECT"
exec ./run.sh "\$@" >> "\$PROJECT/build/app.log" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/PenPlotStudio"

# ---- metadata -------------------------------------------------------------
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$PROJECT/penplot/__init__.py" | head -1)"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$NAME</string>
    <key>CFBundleDisplayName</key><string>$NAME</string>
    <key>CFBundleExecutable</key><string>PenPlotStudio</string>
    <key>CFBundleIdentifier</key><string>se.penplot.studio</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION:-1.0}</string>
    <key>CFBundleVersion</key><string>${VERSION:-1.0}</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
</dict>
</plist>
PLIST
printf 'APPL????' > "$APP/Contents/PkgInfo"

touch "$APP"
echo "built: $APP"
