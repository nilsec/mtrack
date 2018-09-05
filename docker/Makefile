TAG="mtrack"

.PHONY: default clean reset
default:
	mkdir ./mtrack/mtrack
	cp -r ../mtrack ./mtrack/mtrack
	cp -r ../tests ./mtrack/mtrack
	cp ../requirements.txt ./mtrack/mtrack
	cp ../setup.py ./mtrack/mtrack
	docker build -t $(TAG) ./mtrack

clean:
	rm -rf ./mtrack/mtrack

reset:
	docker rm -f $(TAG)
