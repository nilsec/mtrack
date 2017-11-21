TAG="mtrack"

.PHONY: default clean push configure
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

clean:
	-docker rmi -f gurobi:latest
	-docker rmi -f $(TAG)

test: default
	docker run --rm $(TAG)

configure: default
	docker run -d -v $(HOST_WORK_DIR):/src/work --name mtrack -i -t $(TAG):latest /bin/bash
	docker exec -ti mtrack sh -c "echo y | grbgetkey $(GUROBI_KEY)"

reset:
	docker rm -f $(TAG)
