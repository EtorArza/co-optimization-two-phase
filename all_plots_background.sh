source venv/bin/activate

python src/jorgenrem_experiment.py --plot &
python src/evogym_experiment.py --plot &
python src/robogrammar_experiment.py --plot &
python src/gymrem2d_experiment.py --plot &


wait
echo "produced all the plots"

# Copy results to paper dir
rsync -zarv --delete --prune-empty-dirs --include '*/' --include '*.pdf' --exclude '*' results/ ../paper/results/

# Recompile paper
current=`pwd`
cd ../paper
pdflatex -synctex=1 -interaction=nonstopmode main.tex
cd $current 