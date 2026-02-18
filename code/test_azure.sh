#!/bin/bash
# Test Azure Blob Storage connection (read + write) dùng curl + SAS token
# Chạy: bash code/test_azure.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

# Load .env
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        key="${key// /}"
        value="${value%"${value##*[![:space:]]}"}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
    done < "$ENV_FILE"
    echo "Loaded .env"
else
    echo "ERROR: .env not found at $ENV_FILE"
    exit 1
fi

# Parse AccountName từ connection string
ACCT=$(echo "$AZURE_CONNECTION_STRING" | grep -oP 'AccountName=\K[^;]+')
CONTAINER="${AZURE_CONTAINER:-ssl4mis-weights}"
SAS="${AZURE_SAS_TOKEN}"

echo ""
echo "=== Azure Config ==="
echo "  Account   : $ACCT"
echo "  Container : $CONTAINER"
echo "  SAS token : ${SAS:0:30}..."
echo ""

if [ -z "$ACCT" ] || [ -z "$SAS" ]; then
    echo "ERROR: Thiếu AZURE_CONNECTION_STRING hoặc AZURE_SAS_TOKEN trong .env"
    exit 1
fi

BASE_URL="https://${ACCT}.blob.core.windows.net/${CONTAINER}"

# ============ TEST 1: WRITE (upload file nhỏ) ============
echo "=== TEST 1: WRITE ==="
TEST_CONTENT="ssl4mis-azure-test-$(date +%s)"
TEST_BLOB="test/connection_test.txt"

HTTP_CODE=$(curl -s -o /tmp/azure_write_resp.txt -w "%{http_code}" -X PUT \
    -H "x-ms-blob-type: BlockBlob" \
    -H "Content-Type: text/plain" \
    -d "$TEST_CONTENT" \
    "${BASE_URL}/${TEST_BLOB}?${SAS}")

if [ "$HTTP_CODE" = "201" ]; then
    echo "  ✓ WRITE OK (HTTP 201)"
else
    echo "  ✗ WRITE FAILED (HTTP $HTTP_CODE)"
    cat /tmp/azure_write_resp.txt
    echo ""
    echo "Kiểm tra lại SAS token permissions (cần: Write, Create, Object)"
    exit 1
fi

# ============ TEST 2: READ (download file vừa upload) ============
echo ""
echo "=== TEST 2: READ ==="
HTTP_CODE=$(curl -s -o /tmp/azure_read_resp.txt -w "%{http_code}" \
    "${BASE_URL}/${TEST_BLOB}?${SAS}")

if [ "$HTTP_CODE" = "200" ]; then
    CONTENT=$(cat /tmp/azure_read_resp.txt)
    if [ "$CONTENT" = "$TEST_CONTENT" ]; then
        echo "  ✓ READ OK - Content matches: '$CONTENT'"
    else
        echo "  ✗ READ content mismatch"
        echo "    Expected: $TEST_CONTENT"
        echo "    Got     : $CONTENT"
    fi
else
    echo "  ✗ READ FAILED (HTTP $HTTP_CODE)"
    cat /tmp/azure_read_resp.txt
fi

# ============ TEST 3: LIST blobs trong container ============
echo ""
echo "=== TEST 3: LIST container ==="
HTTP_CODE=$(curl -s -o /tmp/azure_list_resp.txt -w "%{http_code}" \
    "${BASE_URL}?restype=container&comp=list&${SAS}")

if [ "$HTTP_CODE" = "200" ]; then
    BLOB_COUNT=$(grep -o '<Name>' /tmp/azure_list_resp.txt | wc -l)
    echo "  ✓ LIST OK - $BLOB_COUNT blob(s) in container"
else
    echo "  ✗ LIST FAILED (HTTP $HTTP_CODE)"
    cat /tmp/azure_list_resp.txt
fi

echo ""
echo "=== SUMMARY ==="
echo "Nếu cả 3 test đều OK → chạy: bash code/run_pp4_tuning.sh"
