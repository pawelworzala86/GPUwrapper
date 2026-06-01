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
const tokenData = new Float32Array(128).fill(0.5)

const wgSizeX = 256;
const wgSizeY = 1;
const wgSizeZ = 1;

const workgroups = {
  x: Math.ceil(COLS / wgSizeX),
  y: Math.ceil(ROWS / wgSizeY),
  z: Math.ceil(VECTORS / wgSizeZ),
}

function getTimestamp(){
    return new Date().getTime()
}





const inputBuffer = gpu.createBuffer(0,'storage', VECTORS * ROWS * COLS, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)
const configDataBuffer = gpu.createBuffer(1,'uniform', 3, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
const tokenDataBuffer = gpu.createBuffer(2,'uniform', 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)



let time = getTimestamp()

gpu.writeBuffer(inputBuffer, input)
gpu.writeBuffer(tokenDataBuffer, tokenData)
gpu.writeBuffer(configDataBuffer, configData)

gpu.begin()
gpu.runShader(workgroups)
gpu.end()

let randomResult = await gpu.getData(inputBuffer, VECTORS * ROWS * COLS)

let end = getTimestamp()
console.log(randomResult.slice(0, 16));
console.log('time: ', end-time);



time = getTimestamp()

gpu.begin()
for(let i=0;i<200;i++){
  //gpu.writeBuffer(inputBuffer, input)
  gpu.writeBuffer(tokenDataBuffer, tokenData)
  gpu.writeBuffer(configDataBuffer, configData)

  gpu.runShader(workgroups)
}
gpu.end()

end = getTimestamp()

randomResult = await gpu.getData(inputBuffer, VECTORS * ROWS * COLS)

console.log(randomResult.slice(0, 16));
//console.log(randomResult.slice(16, 32));
console.log('time: ', end-time);











const gpu2 = new GPU('test.wgsl')





const inputBuffer2 = gpu2.createBuffer(0,'storage', VECTORS * ROWS * COLS, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, inputBuffer)
const configDataBuffer2 = gpu2.createBuffer(1,'uniform', 3, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
const tokenDataBuffer2 = gpu2.createBuffer(2,'uniform', 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)



time = getTimestamp()

//gpu2.writeBuffer(inputBuffer2, input)
gpu2.writeBuffer(tokenDataBuffer2, tokenData)
gpu2.writeBuffer(configDataBuffer2, configData)

gpu2.begin()
gpu2.runShader(workgroups)
gpu2.end()

let randomResult2 = await gpu2.getData(inputBuffer2, VECTORS * ROWS * COLS)

end = getTimestamp()
console.log(randomResult2.slice(0, 16));
console.log('time: ', end-time);