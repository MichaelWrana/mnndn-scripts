#!/bin/bash
set -e


export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

cd ./website-attacks/rf/

function pyenv_virtualenv_exists {
    pyenv virtualenvs --bare | grep -q "^rfenv$"
}

if pyenv_virtualenv_exists; then
    echo "Step 1: Virtual environment 'rfenv' already exists. Activating it..."
else
    echo "Step 1: Creating a pyenv virtual environment named 'rfenv' with Python 3.6.15"
    pyenv virtualenv 3.6.15 rfenv
fi


echo "Step 2: Activate the 'rfenv' virtual environment"
pyenv activate rfenv

echo "Step 3: Install dependencies from requirements.txt"
pip install -r requirements.txt

echo "Step 4: Run the script rf_extract.py"
python rf_extract.py

echo "Step 5: Move generated .npy files to the target directory"
TARGET_DIR="./proj-RF/RF/dataset"
# mkdir -p "$TARGET_DIR"
mv *.npy "$TARGET_DIR"/


echo "Step 6: Training the model"
cd ./proj-RF/RF
python train.py

echo "Step 7: Testing the model"
python test.py

echo "Step 8: Close Virtual Env"
source deactivate