p2prpc generate-broker function.py
sudo docker-compose -f broker/broker.docker-compose.yml build
sudo docker-compose -f broker/broker.docker-compose.yml up -d

# TODO maybe I don't need this
output=$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' broker-discovery);
while [[ "$output" == "" ]]; do
  output=$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' broker-discovery);
  echo $$output;
  sleep 1;
done;
echo $$output:5002 > 'discovery.txt';
cat 'discovery.txt';