TAG="mtrack"

.PHONY: default clean reset
default:
	mkdir ./mtrack/mtrack
	cp -r ../mtrack ./mtrack/mtrack
	cp ../requirements.txt ./mtrack/mtrack
	cp ../setup.py ./mtrack/mtrack
	docker build -t $(TAG) ./mtrack

clean:
	rm -rf ./mtrack/mtrack
	-docker rmi -f mtrack:latest
	-docker rmi -f $(TAG)

reset:
	docker rm -f $(TAG)
