# DIRNG analysis and Trevisan extractor server

This repository holds the software used by [CURBy](https://random.colorado.edu)
to analyze data from the NIST looophole-free Bell experiment and to extract randomness
according to the Device-independent randomness generation protocol.

## Extractor Server

The extractor is a Docker container that hosts a ZMQ server (written in Python 3).

### Setup

First build the Docker image: `docker build . -t trevisan`. To run use `docker-compose up` or `docker-compose up -d` to run in the background. To stop, type `docker-compose down`.

The server expects a JSON string as input. The format is as follows:

```jsonc
{
   "cmd": "command",
   "params": {} // dictionary of input parameters
}
```

The available commands are:

*Calculate the entropy in the data and determine whether the entropy exceeds the required number of bits for the protocol to succeed.*

```js
{
   "cmd": "process_entropy",
   "params": {
      data, // base64 representation of the raw data encoded as a UTF-8 string,
      epsilonBias, // float encoded as UTF-8 representing the bias in the settings distribution,
      beta, // float encoded as UTF-8,
      pefs: // 4x4 nested array of floats encoded to UTF-8, for example...
         [[0.9999999999999982 1.006510090058448  1.0069115515256613 0.9019520376479409],
         [1.0000000000000266 0.9289727143471139 0.9930607468650423 1.033425543789122 ],
         [0.9999999999999998 0.993463817396224  0.9275174893374223 1.032381772978263 ],
         [0.9999999999999745 0.9594734203226882 0.9609260606280244 1.0317220938924305]],
      nBitsThreshold, // float encoded as UTF-8 that represents the number of bits of entropy needed for success
      errorSmoothness, // float encoded as UTF-8 that is the error level for the data. Typically 0.8*2^(-64)
      isQuantum, // boolean. If true it meens we are computing and using QEFs instead of PEFs
   }
}
```

output from 'process_entropy':

```js
{
   entropy, // float encoded as UTF-8 of the computed number of bits of entropy present in the data
   isThereEnoughEntropy // boolean as to whether there is enough entropy to run the extractor
}
```

*Run the Trevisan extractor on the outcome bits from Alice and Bob to extract the certified output bits*

```js
{
   cmd: 'extract',
   params: {
      data, // base64 representation of the raw data encoded as a UTF-8 string,
      seed, // base64 representation of seed
      entropy, // float encoded as UTF-8 that represents the number of bits of entropy in the inputs to the extractor,
      nBitsOut, // int encoded as UTF-8 that represents the number of output bits from the extractor (typically 512),
      errorExtractor, // float encoded as UTF-8 that is the error level for the extractor. Typically 0.2*2^(-64),
      stoppingCriteria, // int encoded as UTF-8; number of trials to look at
   }
}
```

output from 'extract':

```js
{
   outBits // base64 encoded output bytes
}
```


*Get the relevant experimental parameters that need to be precommitted too. This is where things like the PEFs and entropy thresholds are computed.*

```js
{
   "cmd": "get_experiment_parameters",
   "params": {
      epsilonBias, // float encoded as UTF-8 representing the bias in the settings distribution,
      nBitsOut, // int encoded as UTF-8 that represents the number of output bits from the extractor (typically 512),
      errorSmoothness, // float encoded as UTF-8 that is the error level for the data. Typically 0.8*2^(-64),
      errorExtractor, // float encoded as UTF-8 that is the error level for the extractor. Typically 0.2*2^(-64),
      isQuantum, // boolean. If true it meens we are computing and using QEFs instead of PEFs
      freq: // 4x4 nested array of ints encoded to UTF-8, for example:
         [[3587826, 66849, 64304, 2000],
            [3644178, 7733, 44673, 21698],
            [3642347, 47090, 6879, 22013],
            [3681573, 10585, 9913, 18796]]
   }
}
```

output from 'get_experiment_parameters':

```js
{
    beta, // float encoded as UTF-8,
    gain, // float encoded as UTF-8 that represents the bits of entropy accumulated per trial,
    nBitsThreshold, // float encoded as UTF-8 that represents the number of bits of entropy needed for success,
    nTrialsNeeded, // int encoded as UTF-8 that is the minimum number of trials needed to reach the nBitsThreshold of entropy for success,
    seedLength, // int encoded as UTF-8 that is the number of seed bits the extractor requires,
    "pefs": // 4x4 nested array of floats encoded to UTF-8, for example:
      [[0.9999999999999982 1.006510090058448  1.0069115515256613 0.9019520376479409],
      [1.0000000000000266 0.9289727143471139 0.9930607468650423 1.033425543789122 ],
      [0.9999999999999998 0.993463817396224  0.9275174893374223 1.032381772978263 ],
      [0.9999999999999745 0.9594734203226882 0.9609260606280244 1.0317220938924305]]
}
```

*Parameters that need to be pre-committed before randomness generation*

- nBitsOut
- nBitsThreshold
- errorSmoothness
- errorExtractor
- epsilonBias
- seedLength
- stoppingCriteria
- beta
- isQuantum
- pefs
