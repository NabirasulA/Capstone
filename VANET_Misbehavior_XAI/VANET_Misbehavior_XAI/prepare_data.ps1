# Script to prepare the Veremi dataset for the VANET Misbehavior XAI project

# Create data directories if they don't exist
New-Item -ItemType Directory -Force -Path "VANET_Misbehavior_XAI\data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "VANET_Misbehavior_XAI\data\processed" | Out-Null

# Copy the Veremi dataset to the project's raw data directory
Write-Host "Copying Veremi dataset to project directory..."
Copy-Item -Path "Veremi_final_dataset.csv" -Destination "VANET_Misbehavior_XAI\data\raw\veremi_dataset.csv" -Force

# Copy the filtered DDoS attacks dataset if it exists
if (Test-Path "ddos_attacks_only.csv") {
    Write-Host "Copying DDoS attacks dataset to project directory..."
    Copy-Item -Path "ddos_attacks_only.csv" -Destination "VANET_Misbehavior_XAI\data\raw\ddos_attacks_only.csv" -Force
}

Write-Host "Data preparation complete. The dataset is ready for use with the VANET Misbehavior XAI project."
Write-Host "You can now run the project using: python main.py --data_path data/raw --config config.yaml"