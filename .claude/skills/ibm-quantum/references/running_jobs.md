# Running jobs on IBM Quantum hardware

Verified against qiskit 2.5.2 and qiskit-ibm-runtime 0.50.0 (October 2026). Use the `ibm-quantum` environment's
python (printed by `scripts/setup_env.sh`).

## Contents

1. Connect with a saved account
2. Validate locally first (no QPU time)
3. Transpile and inspect the circuit
4. Estimate before spending
5. Submit: job mode and batch mode
6. Error suppression options
7. Read results and record usage
8. Plan facts that shape these choices

## 1. Connect with a saved account

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(name="open")   # a saved account by name; never a key in code or notebooks
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=24)
# or a specific device: backend = service.backend("ibm_marrakesh")
```

The API key lives only in `~/.qiskit/qiskit-ibm.json`. Code that is committed or shared should load accounts by
name, so it runs on any machine where that name has been saved, and the key never reaches a repository.

## 2. Validate locally first (no QPU time)

Fake backends carry real device properties (coupling map, basis gates, error rates), and AerSimulator can run
them with that noise model. Use them to debug circuits and estimate how noise affects the result before using
any minutes.

```python
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh  # Heron r2, 156 qubits; also FakeFez, FakeTorino, FakeMiami

sim = AerSimulator.from_backend(FakeMarrakesh())
```

Exact simulation of a noisy 156-qubit device is not possible. For more than about 30 active qubits, use
`AerSimulator(method="matrix_product_state")`, or validate on a smaller subproblem.

## 3. Transpile and inspect the circuit

```python
from qiskit.transpiler import generate_preset_pass_manager

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
isa_circuit = pm.run(circuit)
two_qubit = sum(1 for instruction in isa_circuit.data if len(instruction.qubits) == 2)
print(f"two-qubit gates: {two_qubit}, depth: {isa_circuit.depth()}")
```

Report the two-qubit gate count to the user. It is the main driver of noise, and it tells them whether the
result is likely to be meaningful before they spend minutes on it.

## 4. Estimate before spending

- **Server-side dry run.** The client-side Sampler can validate a job and estimate its usage without running it
  on a QPU: `Sampler(mode=backend).run(pubs, dry_run=True)`. The results contain random mock data. Access may be
  restricted on some plans, so fall back to the next two methods if it is refused.
- **Back-of-envelope.** QPU time ≈ shots × (repetition delay + circuit duration) per circuit, plus a few seconds of
  overhead per job. The repetition delay is `backend.configuration().default_rep_delay`, 250 µs on current
  Heron devices, so 10,000 shots take about 3 s.
- **A tiny real job first** on a new backend or circuit type, for example 1,000 shots of one circuit. Then read
  `job.usage()` and scale the estimate from that.

## 5. Submit: job mode and batch mode

```python
from qiskit_ibm_runtime import Batch
from qiskit_ibm_runtime.executor_sampler import Sampler  # client-side Sampler; SamplerV2 is deprecated in 0.50

# Job mode: one job.
sampler = Sampler(mode=backend)
job = sampler.run([isa_circuit], shots=10_000)
print("job id:", job.job_id())

# Batch mode: many independent jobs scheduled together. Open Plan allows batch but not sessions.
with Batch(backend=backend) as batch:
    sampler = Sampler(mode=batch)
    jobs = [sampler.run([c], shots=10_000) for c in isa_circuits]
```

`SamplerV2` still works, and existing code using it needs no change yet. It is deprecated as of 0.50.0, and its
removal comes no sooner than three months after that release.

## 6. Error suppression options

```python
sampler.options.twirling.enable_gates = True      # Pauli twirling of two-qubit gates
sampler.options.twirling.enable_measure = True    # readout twirling
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XpXm"
```

- Twirling runs randomised copies of each circuit and spreads the requested shots across them. The
  `num_randomizations` and `shots_per_randomization` options control the split, and both are "auto" by default.
  Check the job's actual usage after the first twirled run.
- For expectation values (EstimatorV2), the resilience options add readout mitigation and zero-noise
  extrapolation. Those cost extra circuits, so include them in the estimate.

## 7. Read results and record usage

```python
result = job.result()
counts = result[0].data.meas.get_counts()     # "meas" is the register created by measure_all()
seconds = job.usage()                          # QPU seconds actually charged
```

Append one line per job to a log in the project, for example `logs/ibm_jobs.jsonl`. Results stay traceable to the
job, device and spend, and the remaining budget can be checked without the dashboard:

```python
import json, datetime
with open("logs/ibm_jobs.jsonl", "a") as log:
    log.write(json.dumps({
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "job_id": job.job_id(), "backend": backend.name, "account": "open",
        "circuits": 1, "shots": 10_000, "qpu_seconds": seconds, "purpose": "H3 ansatz comparison, Apulia p=1",
    }) + "\n")
```

## 8. Plan facts that shape these choices

These are from IBM's plan documentation as of October 2026; check https://quantum.cloud.ibm.com/docs/guides/plans-overview
when in doubt.

- **QPU minutes.** The Open Plan gives up to 10 QPU minutes per 28-day rolling window. A 2026 promotion adds 180
  minutes over 12 months for active users.
- **Region.** Open Plan instances are in us-east only.
- **Execution modes.** Open Plan users cannot submit session jobs; job and batch modes work.
- **Paid plans.** Flex, Pay-As-You-Go and Premium allow sessions. Flex minutes must be allocated to an instance
  before they can be used.
