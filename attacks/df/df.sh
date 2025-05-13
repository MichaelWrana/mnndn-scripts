#!/bin/bash
set -e


export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

cd ./website-attacks/df/

function pyenv_virtualenv_exists {
    pyenv virtualenvs --bare | grep -q "^dfenv$"
}

if pyenv_virtualenv_exists; then
    echo "Step 1: Virtual environment 'dfenv' already exists. Activating it..."
else
    echo "Step 1: Creating a pyenv virtual environment named 'dfenv' with Python 3.6.15"
    pyenv virtualenv 3.6.15 dfenv
fi


echo "Step 2: Activate the 'dfenv' virtual environment"
pyenv activate dfenv

echo "Step 3: Install dependencies from requirements.txt"
pip install -r requirements.txt

echo "Step 4: Run the script df-extract.py"
python df-extract.py

echo "Step 5: Move generated .pkl files to the target directory"
TARGET_DIR="./deep-fingerprinting-test/df/dataset/ClosedWorld/NoDef"
mkdir -p "$TARGET_DIR"
mv *.pkl "$TARGET_DIR"/

echo "Step 6: Run the script ClosedWorld_DF_NoDef.py"
python ./deep-fingerprinting-test/df/src/ClosedWorld_DF_NoDef.py


echo "Step 7: Close Virtual Env"
source deactivate