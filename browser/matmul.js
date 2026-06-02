import { GPU, setDevice } from './gpu.js';

//Object.assign(globalThis, globals);

//const navigator = { gpu: create([]) };

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});

setDevice(device);

// ----- rozmiary macierzy -----
const M = 1024; // rows A, C
const K = 1024; // cols A, rows B
const N = 1024; // cols B, C

// A: M x K, B: K x N, C: M x N
const A = new Float32Array(M * K);
const B = new Float32Array(K * N);
const C = new Float32Array(M * N);

// prosta inicjalizacja
for (let i = 0; i < A.length; i++) A[i] = Math.random();
for (let i = 0; i < B.length; i++) B[i] = Math.random();

const dimsData = new Uint32Array([M, N, K]);

// ----- GPU setup -----
const gpu = await GPU.create('matmul.wgsl');

console.log('shader loaded');

const bytesA = A.length;
const bytesB = B.length;
const bytesC = C.length;
const bytesDims = dimsData.length;

const bufferA = gpu.createBuffer({
  binding: 0,
  type: 'storage',
  size: bytesA,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const bufferB = gpu.createBuffer({
  binding: 1,
  type: 'storage',
  size: bytesB,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const bufferC = gpu.createBuffer({
  binding: 2,
  type: 'storage',
  size: bytesC,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

const bufferDims = gpu.createBuffer({
  binding: 3,
  type: 'uniform',
  size: bytesDims,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

console.log('buffers created');

// write data
gpu.writeBuffer(bufferA, A);
gpu.writeBuffer(bufferB, B);
gpu.writeBuffer(bufferC, C); // zero/garbage, nieważne
gpu.writeBuffer(bufferDims, dimsData);

console.log('buffers setted');

// workgroup size = (16,16,1) => global grid
const wgSizeX = 16;
const wgSizeY = 16;

const workgroups = {
  x: Math.ceil(N / wgSizeX),
  y: Math.ceil(M / wgSizeY),
  z: 1,
};

function nowMs() {
  return performance.now();
}

// ----- GPU benchmark -----
let t0 = nowMs();
console.log('t0:', t0);
gpu.begin();
gpu.runShader(workgroups);
gpu.end();
let t1 = nowMs();
console.log('t1:', t1);

const resultGPU = await gpu.getData(bufferC, bytesC);
let t2 = nowMs();
console.log('t2:', t2);

console.log('GPU C[0..15]:', resultGPU.slice(0, 16));
console.log('GPU compute time (ms):', t1 - t0);
console.log('GPU total (compute + readback) (ms):', t2 - t0);

// ----- CPU matmul (TS) -----
function matmulCPU(
  A,
  B,
  M,
  K,
  N,
) {
  const C = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = 0.0;
      for (let k = 0; k < K; k++) {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = acc;
    }
  }
  return C;
}

t0 = nowMs();
const resultCPU = matmulCPU(A, B, M, K, N);
t1 = nowMs();

console.log('CPU C[0..15]:', resultCPU.slice(0, 16));
console.log('CPU time (ms):', t1 - t0);

// prosta weryfikacja różnicy
let maxDiff = 0;
for (let i = 0; i < 100; i++) {
  const d = Math.abs(resultCPU[i] - resultGPU[i]);
  if (d > maxDiff) maxDiff = d;
}
console.log('max diff (first 100 elems):', maxDiff);