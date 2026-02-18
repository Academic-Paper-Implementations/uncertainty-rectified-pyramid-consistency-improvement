#!/bin/bash
# Chạy script này trên Lightning AI để diagnose vấn đề
# bash code/diagnose.sh

echo "=== Python location ==="
which python
python --version

echo ""
echo "=== File location ==="
ls -la /teamspace/studios/this_studio/code/khangpx_improvement.py

echo ""
echo "=== First 15 lines of file ==="
head -15 /teamspace/studios/this_studio/code/khangpx_improvement.py

echo ""
echo "=== Check boundary_weight in file ==="
grep -n "boundary_weight" /teamspace/studios/this_studio/code/khangpx_improvement.py | head -5

echo ""
echo "=== Check if file has PP4 docstring ==="
grep -n "PP4\|Boundary-Aware" /teamspace/studios/this_studio/code/khangpx_improvement.py | head -5

echo ""
echo "=== Python sys.path ==="
python -c "import sys; [print(p) for p in sys.path]"

echo ""
echo "=== Try running with --help ==="
python /teamspace/studios/this_studio/code/khangpx_improvement.py --help 2>&1 | grep -E "boundary|root_path|ema"
