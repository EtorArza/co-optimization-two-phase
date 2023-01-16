import sys

if sys.executable.split('/')[-3] != 'venv':
    print("This script requires that conda is deactivated and the python environment in other_repos/RoboGrammar/venv/bin/activate is activated. To achieve this, run the following: \n\nconda deactivate\nsource other_repos/RoboGrammar/venv/bin/activate")
    print("\n\nOnce 'venv' has been loaded, rerun this script.")
