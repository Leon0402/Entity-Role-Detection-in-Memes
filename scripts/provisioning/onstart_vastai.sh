cd /workspace
git clone git@github.com:Leon0402/Entity-Role-Detection-in-Memes.git
cd ./Entity-Role-Detection-in-Memes

python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
poetry install

poetry run python -m meme_entity_detection.scripts.download_data --download-url "https://drive.usercontent.google.com/download?id=1wR90q3N0Vafjl-uDkFl5f8qNWmQEVdAY&export=download&confirm=t" --output-path "./data"

apt install nano htop pciutils 
pipx install nvitop

# echo 'set -g mouse on' >> ~/.tmux.conf