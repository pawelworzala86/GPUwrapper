import { create, globals } from 'webgpu'
import { GPU, setDevice } from './gpu.ts'

Object.assign(globalThis, globals)
const navigator = { gpu: create([]) }

const adapter  = await navigator.gpu.requestAdapter()
const device = await adapter!.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter!.limits.maxBufferSize,
  }
})

setDevice(device)


const gpu = new GPU('random.wgsl')




const VECTORS = 16;
const ROWS = 1024;
const COLS = 1024;

const input = new Float32Array(VECTORS * ROWS * COLS).fill(0.0);
const configData = new Uint32Array([COLS, ROWS, VECTORS]);

const wgSizeX = 8;
const wgSizeY = 8;
const wgSizeZ = 4;

const workgroups = {
  x: Math.ceil(COLS / wgSizeX),
  y: Math.ceil(ROWS / wgSizeY),
  z: Math.ceil(VECTORS / wgSizeZ),
}

function getTimestamp(){
    return new Date().getTime()
}





const inputBuffer = gpu.createBuffer('storage', VECTORS * ROWS * COLS, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)
const configDataBuffer = gpu.createBuffer('uniform', 3, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)




let time = getTimestamp()

gpu.writeBuffer(inputBuffer, input)
gpu.writeBuffer(configDataBuffer, configData)

gpu.runShader(workgroups)

let randomResult = await gpu.getData(inputBuffer, VECTORS * ROWS * COLS)

let end = getTimestamp()
console.log(randomResult.slice(0, 16));
console.log('time: ', end-time);



time = getTimestamp()

//gpu.writeBuffer(inputBuffer, input)
//gpu.writeBuffer(configDataBuffer, configData)

gpu.runShader(workgroups)

end = getTimestamp()

randomResult = await gpu.getData(inputBuffer, VECTORS * ROWS * COLS)

console.log(randomResult.slice(0, 16));
console.log('time: ', end-time);