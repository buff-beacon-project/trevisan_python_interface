# Device-independent randomness generation analysis and Trevisan extractor server

This repository holds the software used by [CURBy](https://random.colorado.edu)
to analyze data from the NIST looophole-free Bell experiment and to extract randomness.

It also contains another docker image to perform a verification of a specified round
of randomness from CURBy-Q.

## Verifier

To perform a verification of a given CURBy-Q round, you will first need [docker and docker-compose](https://www.docker.com/) installed. (You can also use other container managers like [podman](https://podman.io/)).

### Setup

To build, use docker compose:

```sh
docker podman compose --profile=verify build
```

### Verification

To run the verification of a specific round (for example round 123), run
a command like this:

```sh
docker compose --profile verify run --rm verify 123
```

This will temporarily boot up the extractor server (described below),
and use the [CURBy JS Library](https://github.com/buff-beacon-project/curby-js-client)
to fetch and verify the necessary data. Then it will run the raw data
through the extractor and ensure the output matches what was reported.

The [index.ts](./verifier/index.ts) file also serves as a demonstration of
how to use the CURBy Library to fetch randomness data.

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
      data: base64 representation of the raw data encoded as a UTF-8 string,
      seed = base64 representation of seed
      entropy = float encoded as UTF-8 that represents the number of bits of entropy in the inputs to the extractor,
      nBitsOut = int encoded as UTF-8 that represents the number of output bits from the extractor (typically 512),
      errorExtractor = float encoded as UTF-8 that is the error level for the extractor. Typically 0.2*2^(-64),
      stoppingCriteria = int encoded as UTF-8; number of trials to look at
   }
}
output from 'extract':
{
    outBits: base64 encoded output bytes
}
```


*Get the relevant experimental parameters that need to be precommitted too. This is where things like the PEFs and entropy thresholds are computed.*

```
{
   cmd: 'get_experiment_parameters',
   params: {
            epsilonBias = float encoded as UTF-8 representing the bias in the settings distribution,
            nBitsOut = int encoded as UTF-8 that represents the number of output bits from the extractor (typically 512),
            errorSmoothness = float encoded as UTF-8 that is the error level for the data. Typically 0.8*2^(-64),
            errorExtractor = float encoded as UTF-8 that is the error level for the extractor. Typically 0.2*2^(-64),
            isQuantum = boolean. If true it meens we are computing and using QEFs instead of PEFs
            freq = 4x4 nested array of ints encoded to UTF-8,
               Example:
               [[3587826, 66849, 64304, 2000],
                [3644178, 7733, 44673, 21698],
                [3642347, 47090, 6879, 22013],
                [3681573, 10585, 9913, 18796]]
    }
}
output from 'get_experiment_parameters':
{
    beta =  float encoded as UTF-8,
    gain = float encoded as UTF-8 that represents the bits of entropy accumulated per trial,
    nBitsThreshold = float encoded as UTF-8 that represents the number of bits of entropy needed for success,
    nTrialsNeeded = int encoded as UTF-8 that is the minimum number of trials needed to reach the nBitsThreshold of entropy for success,
    seedLength = int encoded as UTF-8 that is the number of seed bits the extractor requires,
    pefs = 4x4 nested array of floats encoded to UTF-8,
               Example:
               [[0.9999999999999982 1.006510090058448  1.0069115515256613 0.9019520376479409],
               [1.0000000000000266 0.9289727143471139 0.9930607468650423 1.033425543789122 ],
               [0.9999999999999998 0.993463817396224  0.9275174893374223 1.032381772978263 ],
               [0.9999999999999745 0.9594734203226882 0.9609260606280244 1.0317220938924305]]
}
```

### Parameters that need to be pre-committed before the data request is made

```
nBitsOut = int that represents the number of output bits from the extractor (typically 512),
nBitsThreshold = float that represents the number of bits of entropy needed for success,
errorSmoothness = float that is the error level for the data. Typically 0.8*2^(-64),
errorExtractor = float that is the error level for the extractor. Typically 0.2*2^(-64),
epsilonBias = float encoded as UTF-8 representing the bias in the settings distribution,
seedLength = int that is the number of seed bits the extractor requires,
stoppingCriteria = int; number of trials to look at (typically 15E6),
beta =  float scaling parameter for the entropy calculation,
isQuantum = boolean. If true it meens we are computing and using QEFs instead of PEFs,
pefs = 4x4 nested array of floats,
       Example:
       [[0.9999999999999982 1.006510090058448  1.0069115515256613 0.9019520376479409],
       [1.0000000000000266 0.9289727143471139 0.9930607468650423 1.033425543789122 ],
       [0.9999999999999998 0.993463817396224  0.9275174893374223 1.032381772978263 ],
       [0.9999999999999745 0.9594734203226882 0.9609260606280244 1.0317220938924305]]
```
