#!/usr/bin/python

from subprocess import call
import shutil

width=1024
height=1024
x=int((32375-300)/2.0)
y=int((26890-300)/2.0)
sz=1420-300

x=int((32375-1300)/2.0)
y=int((26890-1300)/2.0)
sz=1420-300
for dz in range(0,500):

  z = sz + dz
  section = str(width) + "_" + str(height) + "/" + str(x) + "_" + str(y) + "_" + str(z)
  filename = "raw_" + "%04d" % dz + ".png"

  call([
      "wget",
      "http://neurocity.janelia.org/dvid/api/node/83/rawdata/raw/xy/" + section,
      "-O",
      filename])

