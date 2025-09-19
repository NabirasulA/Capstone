# Script to set up the Python environment for the VANET Misbehavior XAI project

# Create a virtual environment
Write-Host "Creating Python virtual environment..."
python -m venv venv

# Activate the virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host "Environment setup complete."
Write-Host "To activate the environment in the future, run: .\venv\Scripts\Activate"
Write-Host "You can now run the project using: python main.py --data_path data/raw --config config.yaml"