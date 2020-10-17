SHELL:=/bin/bash

.PHONY: services delete checkservices test evaluatedbs logbroker
THIS_FILE := $(lastword $(MAKEFILE_LIST))

buildandstart:
	cd P2P-RPC/; python setup.py install; cd ..
	p2prpc generate-broker function.py
	sudo docker-compose -f broker/broker.docker-compose.yml build
	sudo docker-compose -f broker/broker.docker-compose.yml up -d

	output=$$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' broker-discovery); \
	while [[ "$$output" == "" ]]; do \
		output=$$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' broker-discovery); \
		echo $$output; \
		sleep 1; \
	done; \
	echo $$output:5002 > 'discovery.txt'; \
	cat 'discovery.txt';

	p2prpc generate-client function.py discovery.txt

	sudo docker-compose -f client/client.docker-compose.yml build
	sudo docker-compose -f client/client.docker-compose.yml up -d
	rm -R client/p2prpc

	p2prpc generate-worker function.py discovery.txt

	sudo docker-compose -f worker/worker.docker-compose.yml build
	sudo docker-compose -f worker/worker.docker-compose.yml up -d
	@$(MAKE) -f $(THIS_FILE) checkservices

startservices:
	sudo docker-compose -f broker/broker.docker-compose.yml up -d
	sudo docker-compose -f client/client.docker-compose.yml up -d
	sudo docker-compose -f worker/worker.docker-compose.yml up -d

delete:
	sudo docker-compose -f broker/broker.docker-compose.yml kill || true
	sudo docker-compose -f broker/broker.docker-compose.yml rm -f || true
	sudo docker-compose -f client/client.docker-compose.yml kill || true
	sudo docker-compose -f client/client.docker-compose.yml rm -f || true
	sudo docker-compose -f worker/worker.docker-compose.yml kill || true
	sudo docker-compose -f worker/worker.docker-compose.yml rm -f || true

	sudo rm -R worker || true
	sudo rm -R broker || true
	sudo rm -R client || true

checkservices:
	@sudo printf 'mongo-client  '; sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-client
	@sudo printf 'mongo-client  ' > 'iplist.txt'; sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-client >> 'iplist.txt'
	@sudo printf 'mongo-broker  '; sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-broker
	@sudo printf 'mongo-broker  ' >> 'iplist.txt'; sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-broker >> 'iplist.txt'
	@sudo printf 'mongo-worker  '; sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-worker
	@sudo printf 'mongo-worker  ' >> 'iplist.txt'; sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-worker >> 'iplist.txt'
	@sudo printf 'broker        '; sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' broker

logbroker:
	sudo docker-compose -f broker/broker.docker-compose.yml logs -f broker

logworker:
	sudo docker-compose -f worker/worker.docker-compose.yml logs -f worker

run:
	@$(MAKE) -f $(THIS_FILE) cleandbs
	@export MONGO_HOST=$(shell sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-client); \
	export MONGO_PORT=27017; \
	export PYTHONPATH=$$PYTHONPATH:./; \
	python main.py
