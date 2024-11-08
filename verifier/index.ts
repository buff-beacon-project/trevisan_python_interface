import { CHAINS, DIRNGClient, type RoundData } from '@buff-beacon-project/curby-client'
import zeromq from 'zeromq'

async function trevisanCommand(msg: any){
  const zmq = new zeromq.Request({
    receiveTimeout: 20000,
    sendTimeout: 20000,
    connectTimeout: 20000,
  })
  zmq.connect('tcp://trevisan:5553')
  print('Sending command to trevisan')
  await zmq.send(JSON.stringify(msg))
  const [bytes] = await zmq.receive()
  const json = (new TextDecoder()).decode(bytes)
  const res = JSON.parse(json)
  if (res.error) {
    throw new Error(`Trevisan Error: ${res.error}`)
  }
  zmq.close()
  return res
}

let grp = false
function print(msg?: string) {
  console.log((grp ? '  ' : '') + msg)
}

function groupStart(name: string) {
  print(`\n== ${name} ==`)
  grp = true
}

function groupEnd() {
  grp = false
  print('')
}

function printRoundInfo(roundData: RoundData){
  const chainCid = `${roundData.chain.cid}`
  const output: any = {
    'Round': roundData.round,
    'Stage': roundData.stage,
    'Chain': chainCid,
    'Request Pulse': `${roundData.pulses.request?.cid}`,
    'Response Pulse': `${roundData.pulses.precommit?.cid}`,
  }

  if (roundData.pulses.result) {
    output['Result (Randomness) Pulse'] = `${roundData.pulses.result?.cid}`
  } else if (roundData.pulses.error) {
    output['Error Pulse'] = `${roundData.pulses.error?.cid}`
  }

  console.table(output)
}

// First get the arguments from the command line
const args = process.argv.slice(2)

// Argument should be a round number
const round = parseInt(args[0])

// Validate the round number
if (isNaN(round)) {
  console.error('Error: The first argument should be a round number')
  process.exit(1)
}

console.log(`Validating round ${round}`)

const url = 'https://random.colorado.edu/api'
// Create a new client
const client = new DIRNGClient({
  url,
  validateSeed: true
})

print('Using server: ' + url)

// fetch the round
const roundData = await client.fetchRound(round)

print('')
// Print the round info
printRoundInfo(roundData)

if (!roundData.isComplete) {
  console.error(`Round ${round} is not complete`)
  console.error(`Round is on stage: ${roundData.stage}`)
  process.exit(1)
}

if (roundData.pulses.error) {
  console.error(`Round ${round} did not generate randomness`)
  console.error(`Error pulse: ${roundData.pulses.error.cid}`)
  console.error(`Error message: ${roundData.pulses.error.value.content.payload.reason}`)
}

// print validations
groupStart('Bell Response')
print(`Bell chain: ${CHAINS.bell}`)
if (roundData.validations.bellResponse?.ok) {
  print('Bell chain has valid response pulse')
  print('The dataHash in bell pulse matches CURBy precommit pulse')
} else if (!roundData.pulses.precommit) {
  console.error('No precommit pulse')
} else {
  console.error(`Bell response pulse check failed: ${roundData.validations.bellResponse!.reason}`)
}
groupEnd()

groupStart('Seed Pulse Time Ordering')
if (roundData.validations.seedOrdering?.ok) {
  print('Seed pulse is valid')
  print(`Seed pulse: ${roundData.validations.seedOrdering!.data?.pulse.cid}`)
  print('Seed pulse has correct ordering')
} else if (roundData.pulses.precommit) {
  console.error(`Seed ordering check failed: ${roundData.validations.seedOrdering!.reason}`)
} else {
  console.error('No precommit pulse')
}
groupEnd()

groupStart('Seed Value')
if (roundData.validations.seed?.ok) {
  print('Seed value is valid')
  print(`Seed begins with: (base64) ${roundData.validations.seed!.data?.bytes.slice(0, 10)}...`)
} else if (roundData.pulses.result) {
  console.error(`Seed verification failed: ${roundData.validations.seed!.reason}`)
} else {
  console.error('No result pulse found')
}
groupEnd()

const requestParams = roundData.pulses.request.value.content.payload.parameters

print('Fetching bell data...')
const bellData = await client.fetchRoundData(roundData).catch((err) => {
  console.error('Error fetching bell data:', err.message)
  process.exit(1)
})
print('done (hash OK)')

groupStart('Hypothesis test')
const hypothesisTest = await trevisanCommand({
  cmd: 'process_entropy',
  params: {
    ...requestParams,
    data: bellData,
  }
})

if (!hypothesisTest.isThereEnoughEntropy) {
  console.error('Hypothesis test failed: Not enough entropy')
  if (roundData.pulses.error?.value.content.payload.reason === 'Insufficient entropy') {
    print('OK: Error pulse has correct reason')
  } else {
    console.error('Error: Incorrect reason in error pulse', roundData.pulses.error?.value.content.payload)
  }
  process.exit(0)
}

print('Hypothesis test passed: Enough entropy')
print(`Entropy: ${hypothesisTest.entropy.toFixed(0)} bits`)

groupEnd()

print('Fetching round parameters...')
const params = await client.fetchRoundParams(roundData)
print('done (hash OK)')

groupStart('Output Randomness')

print('Extracting randomness...')
const extractionResult = await trevisanCommand({
  cmd: 'extract',
  params: {
    ...params,
    data: bellData,
  }
})

const claimedRandomness = Buffer.from(roundData.randomness?.bytes()!)
// convert to base64
const claimedRandomnessBase64 = claimedRandomness.toString('base64')

// compare
if (extractionResult.outBits !== claimedRandomnessBase64) {
  console.error('Error: Randomness does not match')
  console.error('Claimed:', claimedRandomnessBase64)
  console.error('Extracted:', extractionResult.outBits)
  process.exit(1)
}

print('Randomness matches')
print(`Randomness: (base64) ${claimedRandomnessBase64}`)
print(`Randomness: (hex) ${claimedRandomness.toString('hex')}`)
groupEnd()

groupStart('Shuffle check')
print('1, 2, 3, 4, 5 becomes...')
const shuffled = roundData.randomness?.shuffled([1, 2, 3, 4, 5])
print(shuffled?.join(', '))
groupEnd()