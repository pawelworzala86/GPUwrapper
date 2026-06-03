import { GPU, setDevice } from './gpu.ts'

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No GPU adapter");

const device = await adapter!.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter!.limits.maxBufferSize,
  }
})

setDevice(device)


const gpu = await GPU.create('random.wgsl')




const VECTORS = 128;
const ROWS = 128;
const COLS = 128;

const TOKENS = 128

const input = new Float32Array(VECTORS * ROWS * COLS).fill(0.0);
const configData = new Uint32Array([COLS, ROWS, VECTORS]);
const tokenData = new Float32Array(TOKENS).fill(0.5)

const wgSizeX = 256;
const wgSizeY = 1;
const wgSizeZ = 1;

const workgroups = {
  x: Math.ceil(TOKENS / wgSizeX),
  y: 1,
  z: 1,
}
/*
const workgroups = {
  x: Math.ceil(COLS / wgSizeX),
  y: Math.ceil(ROWS / wgSizeY),
  z: Math.ceil(VECTORS / wgSizeZ),
}
*/

function getTimestamp(){
    return new Date().getTime()
}





const inputBuffer = gpu.createBuffer({
  binding: 0, 
  type: 'storage', 
  size: VECTORS * ROWS * COLS, 
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
})
const configDataBuffer = gpu.createBuffer({
  binding: 1,
  type: 'uniform',
  size: 3,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})
const tokenDataBuffer = gpu.createBuffer({
  binding: 2,
  type: 'uniform',
  size: 128,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})



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











const gpu2 = await GPU.create('test.wgsl')






const inputBuffer2 = gpu2.createBuffer({
  binding: 0,
  type: 'storage',
  size: VECTORS * ROWS * COLS,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  buffer: inputBuffer, // Reuse the same buffer
})
const configDataBuffer2 = gpu2.createBuffer({
  binding: 1,
  type: 'uniform',
  size: 3,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})
const tokenDataBuffer2 = gpu2.createBuffer({
  binding: 2,
  type: 'uniform',
  size: 128,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})


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