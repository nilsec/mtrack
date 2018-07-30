TAG="mtrack"

.PHONY: default clean push configure test
default:
	mkdir ./mtrack/mtrack
	cp -r ../mtrack ./mtrack/mtrack
	cp ../requirements.txt ./mtrack/mtrack
	cp ../setup.py ./mtrack/mtrack
	docker build -t $(TAG) ./mtrack

push:
	docker push nilsec/mtrack:latest
	docker push nilsec/$(TAG)

clean:
	rm -rf ./mtrack/mtrack
	-docker rmi -f mtrack:latest
	-docker rmi -f $(TAG)

test:
	./run_docker -d $(TAG) python -u ../tests/test_suite.py

reset:
	docker rm -f $(TAG)
