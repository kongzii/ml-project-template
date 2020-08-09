.PHONY: stop component clean tensorboard mlflow jupyter

stop:
	docker-compose stop

dataset:
	clear
	docker-compose build component
	docker-compose run component \
		python3 /app/src/components/dataset/main.py 

train:
	clear
	docker-compose build component
	docker-compose run component \
		python3 /app/src/components/train/main.py 

test:
	clear
	docker-compose build component
	docker-compose run component \
		python3 /app/src/components/test/main.py 

serve:
	clear
	docker-compose up \
		--build \
		--attach-dependencies \
		--abort-on-container-exit \
		serve 

clean:
	docker-compose stop
	docker-compose run clean

tensorboard:
	clear
	docker-compose up \
		--build \
		--attach-dependencies \
		--abort-on-container-exit \
		tensorboard

mlflow:
	clear
	docker-compose up \
		--build \
		--attach-dependencies \
		--abort-on-container-exit \
		mlflow

jupyter:
	clear
	docker-compose up \
		--build \
		--attach-dependencies \
		--abort-on-container-exit \
		jupyter
