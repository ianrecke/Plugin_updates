#!/bin/bash

docker rm -vf "$(docker ps -aq)"
docker rmi "$(docker images -f "dangling=true" -q)"
docker images