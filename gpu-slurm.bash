#SBATCH --account=rrg-pbellec
#SBATCH --time=00:05:00           # time (DD-HH:MM)
#SBATCH --job-name=gpu-test       # job name
#SBATCH --gres=gpu:v100:1         # Number of Nvidia 12G V100 (per node)
#SBATCH --cpus-per-task=4         # CPU cores/threads per GPU request
#SBATCH --mem=5G	          # memory per node
#SBATCH --output=%x_%N_%j.out         # %x job name, %N for node name, %j for jobID

module load singularity
nvidia-smi
singularity exec --nv --no-home ~/projects/rrg-pbellec/CONTAINERS/deep-neuro-docker-gpu.simg python3 keras_test.py
