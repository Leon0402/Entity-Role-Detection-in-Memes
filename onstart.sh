cd /workspace

python3 -m venv work-venv
source /work-venv/bin/activate

touch ./ssh_tmp
echo "-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACA7AfMBv9MYYwSqnohY6il7rkQGWBnMDx+H3KL2x18ApQAAAJBR15eEUdeX
hAAAAAtzc2gtZWQyNTUxOQAAACA7AfMBv9MYYwSqnohY6il7rkQGWBnMDx+H3KL2x18ApQ
AAAEA/uTbVGBoNz/Fb8+RAd2/sIhrK9O8LLAwX/4NpJtli4DsB8wG/0xhjBKqeiFjqKXuu
RAZYGcwPH4fcovbHXwClAAAACXRvbUBuaXhvcwECAwQ=
-----END OPENSSH PRIVATE KEY-----" > ssh_tmp
chmod -R 700 ./ssh_tmp

git clone -c core.sshCommands="ssh -i ./ssh_tmp -o 'ConnectTimeout 3' -o 'StrictHostKeyChecking no' -o 'UserKnownHostsFile /dev/null' '$@' " git@github.com:Leon0402/Entity-Role-Detection-in-Memes.git
cd ./Entity-Role-Detection-in-Memes/

pip install poetry
poetry install

python -m meme_entity_detection.scripts.download_data --download-url "https://drive.usercontent.google.com/download?id=1wR90q3N0Vafjl-uDkFl5f8qNWmQEVdAY&export=download&confirm=t" --output-path "./data"


