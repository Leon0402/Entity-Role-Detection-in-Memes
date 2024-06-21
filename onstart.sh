cd /workspace

python3 -m venv work-venv
source /work-venv/bin/activate

git clone git@github.com:Leon0402/Entity-Role-Detection-in-Memes.git

cd ./Entity-Role-Detection-in-Memes

pip install poetry
poetry install

python -m meme_entity_detection.scripts.download_data --download-url "https://drive.usercontent.google.com/download?id=1wR90q3N0Vafjl-uDkFl5f8qNWmQEVdAY&export=download&confirm=t" --output-path "./data"


