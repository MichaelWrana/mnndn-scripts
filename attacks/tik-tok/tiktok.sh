#!/bin/bash
set -e


export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

cd ./website-attacks/tik-tok/

function pyenv_virtualenv_exists {
    pyenv virtualenvs --bare | grep -q "^tiktokenv$"
}

if pyenv_virtualenv_exists; then
    echo "Step 1: Virtual environment 'tiktokenv' already exists. Activating it..."
else
    echo "Step 1: Creating a pyenv virtual environment named 'tiktokenv' with Python 3.6.15"
    pyenv virtualenv 3.6.15 tiktokenv
fi


echo "Step 2: Activate the 'tiktokenv' virtual environment"
pyenv activate tiktokenv

echo "Step 3: Install dependencies from requirements.txt"
pip install -r requirements.txt

echo "Step 4: Run the script tik_tok_extract.py"
python tik_tok_extract.py

echo "Step 5: Move generated .pkl files to the target directory"
TARGET_DIR="./Tik_Tok/Timing_Features/save_data/Undefended"
mkdir -p "$TARGET_DIR"
mv *.pkl "$TARGET_DIR"/

echo "Step 6: Run the script Tik_Tok_timing_features.py"
cd Tik_Tok/Timing_Features/
python Tik_Tok_timing_features.py Undefended


echo "Step 7: Close Virtual Env"
source deactivate