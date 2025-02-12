## Development Setup

After cloning the repository:

1. Create and activate a Python 3.11 conda environment:
   ```bash
   conda create -n ymir python=3.11
   conda activate ymir
   ```

2. Install Poetry using pip:
   ```bash
   pip install poetry
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Set up pre-commit hooks:
   ```bash
   poetry run pre-commit install
   poetry run pre-commit install --hook-type post-checkout --hook-type post-merge
   ```
