CONTAINER_NAME = gptino
PROJECT_DIR = ~/repos/GPTino

build-docker:
	docker build --platform linux/amd64 -t $(CONTAINER_NAME):latest .
	docker save $(CONTAINER_NAME):latest -o $(CONTAINER_NAME).tar

send:
	rsync -avz --progress $(CONTAINER_NAME).tar m-ben-salah@172.18.47.81:${PROJECT_DIR}/

build-and-send: build-docker send

build-apptainer:
	apptainer build $(CONTAINER_NAME).sif docker-archive://$(CONTAINER_NAME).tar