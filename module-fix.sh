WSL2_VERSION=$(uname -r)
echo "WSL2_VERSION = $WSL2_VERSION"

WSL2_LINK="/lib/modules/$WSL2_VERSION"
if [ -L "${WSL2_LINK}" ]; then
    if [ -e "${WSL2_LINK}" ]; then
        echo "Good link"
        exit 0
    else
        echo "Broken link"
        rm "${WSL2_LINK}"
    fi
elif [ -e "${WSL2_LINK}" ]; then
    echo "Not a link"
    exit 1
else
    echo "Missing"
fi

shopt -s nullglob
for filename in /lib/modules/*; do
    echo "$filename"
    if [ -z "$HEADERS_DIR" ]; then
        HEADERS_DIR="$filename"
    else
        echo "HEADERS_DIR already set to $HEADERS_DIR, fail"
        exit 1
    fi
done

if [ -n "$HEADERS_DIR" ]; then
    echo "Create symbolic link $WSL2_LINK => $HEADERS_DIR"
    ln -s "$HEADERS_DIR" "$WSL2_LINK"
fi
