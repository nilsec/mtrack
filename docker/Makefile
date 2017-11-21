TAG="mtrack"

.PHONY: default clean push
default:
	docker build -t gurobi:latest ./gurobi
	-rm -rf ./gurobi/mtrack/mtrack
	mkdir ./gurobi/mtrack/mtrack
	cp -r ../mtrack ./gurobi/mtrack/mtrack
	cp ../requirements.txt ./gurobi/mtrack/mtrack
	cp ../setup.py ./gurobi/mtrack/mtrack
	docker build -t $(TAG) ./gurobi/mtrack

push: default
	docker push nilsec/gurobi:latest
	docker push nilsec/$(TAG)

clean: default
	-docker rmi -f gurobi:latest
	-docker rmi -f $(TAG)

test: default
	docker run --rm $(TAG)
