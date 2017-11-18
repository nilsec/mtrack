TAG="mtrack"

.PHONY: default
default:
	-docker rmi -f gurobi:latest
	docker build -t gurobi:latest ./gurobi
	-docker rmi -f $(TAG)
	-rm -rf ./gurobi/mtrack/mtrack
	mkdir ./gurobi/mtrack/mtrack
	cp -r ../mtrack ./gurobi/mtrack/mtrack
	cp ../requirements.txt ./gurobi/mtrack/mtrack
	cp ../setup.py ./gurobi/mtrack/mtrack
	docker build -t $(TAG) ./gurobi/mtrack
