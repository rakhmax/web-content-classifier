# Web content context analyzer

## Data preparation
If you want to use dataset of this project just enter:
```bash
$ python3 ./src/preprocess_data.py
```
It takes a while because scrapping is in process.

If you want to use your own dataset create `urls.csv` file inside `data` folder with the following data structure:

| url        |     content     | category | context #1 | ...context |
|------------|-----------------|----------|------------|------------|
| https://.. | Lorem ipsum ..  | Adult    | 0          | 1          |
| http://..  | Lorem ipsum ..  | Sport    | 1          | 0          |
| https://.. | Lorem ipsum ..  | Science  | 1          | 1          |

Values in **url** column are not required, but the column itself should be the 1st.

There should be at least 2 context columns. **1st context column must be the 4th in csv.**

## Installation
To install all dependencies (using venv) and train models do the following:
```bash
$ ./pipeline.sh
```

## How to use

To predict category and context enter:
```bash
$ python3 ./src/predictor.py --url URL [--user USER]
```

`URL` must start with `http(s)://`

For example:
```bash
$ python3 ./src/predictor.py --url https://engadget.com --user office
```

The commard prints predicted values and `True` if office is in context else `False`.