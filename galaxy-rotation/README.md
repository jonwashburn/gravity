# Galaxy Rotation Package

Reproduce:
1. python3 -m pip install -r env/requirements.txt
2. python3 scripts/build_sparc_master_table.py
3. python3 scripts/ledger_final_combined.py

Compile paper:
cd paper && pdflatex -interaction=nonstopmode -halt-on-error dark-matter-galaxy-rotation.tex
