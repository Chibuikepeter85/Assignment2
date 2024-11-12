#! /bin/bash
# ====================================
#SBATCH --job-name=tl_text
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=output.txt
#SBATCH --error=error.txt 
# ====================================

# Activate conda environment
eval "$(~/software/miniforge3/bin/conda shell.bash hook)"
conda activate pytorch_env

# Change to the directory containing the script
cd /home/chibuike.ohanu/work/garbage_transfer_learning

# Run the Python script
python -u Assignment_2.py

