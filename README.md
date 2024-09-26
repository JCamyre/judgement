# judgement

## Environment Set-Up

1. Clone repo
2. In root directory, run `pipenv shell`
   1. If you don't have `pipenv` installed, install with `pip install pipenv`
3. Run `pipenv install`, which will install the packages and use the Python version specified in the Pipfile
4. Create `.env` file in root directory, adding all secret keys and setting the Python path to the root directory by adding `PYTHONPATH="."`

Whenever you open a new terminal, access your pipenv environment via `pipenv shell`
