CONTAINER_NAME = train_gym
PROJECT_DIR = ~/repos/GPTino

build-docker:
	docker build --platform linux/amd64 -t $(CONTAINER_NAME):latest .
	docker save $(CONTAINER_NAME):latest -o $(CONTAINER_NAME).tar

send:
	rsync -avz --progress $(CONTAINER_NAME).tar m-ben-salah@172.18.47.92:${PROJECT_DIR}/

build-and-send: build-docker send

build-apptainer:
	sbatch --job-name=build-apptainer \
		--time=1:00:00 \
		--mem=24G \
		--cpus-per-task=12 \
		--partition=ai \
		--wrap="apptainer build $(CONTAINER_NAME).sif docker-archive://$(CONTAINER_NAME).tar"


run-interactive:
	srun --job-name=run-interactive \
		--time=1:00:00 \
		--mem=24G \
		--cpus-per-task=12 \
		--nodes=2 \
		--partition=ai \
		--gres=gpu:h100:2 \
		--pty apptainer shell --nv $(CONTAINER_NAME).sif