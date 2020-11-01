p2prpc generate-worker function.py discovery.txt
sudo docker-compose -f worker/worker.docker-compose.yml build
sudo docker-compose -f worker/worker.docker-compose.yml up -d
