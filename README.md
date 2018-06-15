Automatic extraction of microtubules in electron microscopy volumes of neural tissue.

1. Start mongodb instance via: mongod --config /etc/mongod.conf

Usage for arbitrary (small scale - no db interaction required) pathwise tracking problems:

0. Install Gurobi and pylp (http://github.com/funkey/pylp) as well as graph-tool(https://graph-tool.skewed.de/). Further requirements will be included. 

1. Extract position (and orientation) for each instance of interest and initialize a 
   G1-Graph where each vertex encodes an instance/candidate (see mtrack/graphs/g1_graph).
   For an example of how to do that and add position and orientation see tests/cases/g1_graph

2. (Optional) If you want to use a custom cost function based on position and orientation infor       mation on vertices, edges and combinations of edges inherit from the g1 graph class and over       write the corresponding methods for calculation of vertex-, edge- and edge-combination cost.

3. Set cost function hyperparemters and pass your initialized g1 graph to solve(...) in mtrack/solve. It will return the solved g1-graph which you can export in e.g. nml format via mtrack/preprocessing/nml_io/g1_to_nml and view with knossos.
