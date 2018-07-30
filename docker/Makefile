TAG="mtrack"

.PHONY: default clean push configure
default:
	mkdir ./mtrack/mtrack
	cp -r ../mtrack ./mtrack/mtrack
	cp ../requirements.txt ./mtrack/mtrack
	cp ../setup.py ./mtrack/mtrack
	docker build -t $(TAG) ./mtrack

push: default
	docker push nilsec/mtrack:latest
	docker push nilsec/$(TAG)

clean:
	rm -rf ./mtrack/mtrack
	-docker rmi -f mtrack:latest
	-docker rmi -f $(TAG)

test: default
	docker run --rm $(TAG)

reset:
	docker rm -f $(TAG)
