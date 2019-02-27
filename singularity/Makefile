TAG="mtrack:v0.1"
TMP_FILE:=$(shell mktemp).img

.PHONY: default push
default:
	sudo singularity build $(TMP_FILE) Singularity
	sudo chown ecksteinn ${TMP_FILE}
