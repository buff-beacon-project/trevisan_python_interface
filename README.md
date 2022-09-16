# bellrand

The Randomness Public Beacon project is an idea created by the National Institute for Standards and Technology (NIST) to help generate truly random binary bits with a security certificate. This beacon will be for public use and is applicable to any scenario which needs unbiased randomness. 
Some potential applications include (but not limited to):
   * Prevention of Gerrymandering
   * Seeding of private randomness sources for communication key generation

The beacon or pipeline has three main components:  
    Experimental  
    Analysis  
    Extractor  

Each phase serves a special purpose. 
The bellrand repository contains all the modules required to realize this public beacon:
* PEF Analysis
* Data Loading
* PEF Calculator
* PEF Accumulator
* Manager
* Main

To run, first build the docker container with the extractor.
```cd extractor ```  
```docker build . -t test ```  
```docker run -it test ```  
```python3 extractor_server.py```

In three other terminals, run ```entropy_server.py```, ```pef_server.py```, and ```manager.py```. then run ```experiment_placeholder.py``` to start.

Necessary Libraries needed to run this software:
* Numba
* Cvxpy
* llvmlite
* Os
* Numpy
* Sympy (to use the seed_length code)
* zmq

