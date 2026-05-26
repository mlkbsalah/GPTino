PROJECT_CONTAINER_NAME = gptino

build-docker:
	docker build --platform linux/amd64 -t $(PROJECT_CONTAINER_NAME):latest .
	docker save $(PROJECT_CONTAINER_NAME):latest -o $(PROJECT_CONTAINER_NAME).tar

send:
	rsync -avz --progress $(PROJECT_CONTAINER_NAME).tar m-ben-salah@172.18.47.81:~/repos/awesome-gpu/

build-and-send: build-docker send

build-apptainer:
	apptainer build $(PROJECT_CONTAINER_NAME).sif docker-archive://$(PROJECT_CONTAINER_NAME).tar