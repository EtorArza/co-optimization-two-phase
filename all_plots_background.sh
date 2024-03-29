source venv/bin/activate

python --version
which python
sleep 2

python src/jorgenrem_experiment.py --plot &
python src/evogym_experiment.py --plot &
python src/robogrammar_experiment.py --plot &
python src/gymrem2d_experiment.py --plot &
python src/jorgenrem_experiment.py --plot_tune &
python src/gymrem2d_experiment.py --plot_tune &


wait
echo "produced all the plots"

# Function to crop all PDF figures in a directory and its subdirectories
crop_pdfs() {
    local directory="$1"

    # Loop through each PDF in the directory and its subdirectories
    find "$directory" -type f -name '*.pdf' | while IFS= read -r pdf; do
        # Construct the output filename (using the same name, thus overwriting the original)
        output="$pdf"

        # Apply pdfcrop
        pdfcrop "$pdf" "$output"

        echo "Cropped $pdf"
    done
}


# Copy results to paper dir
rsync -zarv --delete --prune-empty-dirs --include '*/' --include '*.pdf' --exclude '*' results/ ../paper/results/

# Recompile paper
current=`pwd`
cd ../paper
crop_pdfs "./results/"


# pdflatex -synctex=1 -interaction=nonstopmode main.tex
cd $current 