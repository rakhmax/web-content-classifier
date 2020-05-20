mkdir ./models

python3 -m venv env
source ./env/bin/activate

pip install -r requirements.txt

python3 ./src/prepare_db.py
python3 ./src/train_categories.py
python3 ./src/train_context.py
