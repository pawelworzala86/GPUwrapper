import fs from 'fs'
import { create, globals } from 'webgpu'
import GPU from './gpu.ts'

Object.assign(globalThis, globals)
const navigator = { gpu: create([]) }

const adapter  = await navigator.gpu.requestAdapter()
const device = await adapter!.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter!.limits.maxBufferSize,
  }
})

const gpu = new GPU(device)



const wgslCodeRandom = fs.readFileSync('./shaders/random.wgsl').toString()


const VECTORS = 16;
const ROWS = 1024;
const COLS = 1024;

const input = new Float32Array(VECTORS * ROWS * COLS).fill(0.0);
const configData = new Uint32Array([COLS, ROWS, VECTORS]);

const wgSizeX = 8;
const wgSizeY = 8;
const wgSizeZ = 4;

const workgroupsX = Math.ceil(COLS / wgSizeX);
const workgroupsY = Math.ceil(ROWS / wgSizeY);
const workgroupsZ = Math.ceil(VECTORS / wgSizeZ);

function getTimestamp(){
    return new Date().getTime()
}

let time = getTimestamp()

gpu.setData([
  {
    binding: 0,
    type: 'storage',
    input: input,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  },
  {
    binding: 1,
    type: 'uniform',
    input: configData,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  },
]);

gpu.runShader(wgslCodeRandom, { workgroupsX, workgroupsY, workgroupsZ });

const randomResult = await gpu.getData(0);

let end = getTimestamp()

console.log(randomResult.slice(0, 16));
console.log('time: ', end-time);