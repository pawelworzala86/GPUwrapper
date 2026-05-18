import fs from 'fs'
import { create, globals } from 'webgpu'
import GPU from './gpu.ts'

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };

const adapter  = await navigator.gpu.requestAdapter();
const device = await adapter!.requestDevice();

const gpu = new GPU(device)

// --- PRZYKŁADOWE DANE Z TRANSFORMERA ---
// TODO: tu podepniesz realne dane z Twojej klasy Transformer
const vocab = 50257
const d_model = 128

const logits = new Float32Array(vocab)        // wyjście z W_out * hLast
const hLast = new Float32Array(d_model)       // ostatni wektor z bloku
const Wout = new Float32Array(vocab * d_model)
const W_in_last = new Float32Array(d_model)   // embedding ostatniego tokena

const target = 42                             // przykładowy target
const lr = 1e-3                               // learning rate

// --- (opcjonalnie) Twój random shader 3D ---
const VECTORS = 16;
const ROWS = 1024;
const COLS = 1024;
const count = VECTORS * ROWS * COLS;

const input = new Float32Array(count).fill(0.0);
const configData = new Uint32Array([COLS, ROWS, VECTORS]);

const wgslCodeRandom = fs.readFileSync('./shaders/random.glsl').toString()

const workgroupsX = Math.ceil(COLS / 8)
const workgroupsY = Math.ceil(ROWS / 8)
const workgroupsZ = Math.ceil(VECTORS / 4)



for(let i=0;i<32;i++){
    // 1) random.glsl (jeśli chcesz go odpalić)
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
    ])

    gpu.runShader(wgslCodeRandom, { workgroupsX, workgroupsY, workgroupsZ })

    // jeśli chcesz wynik:
    // const randomResult = await gpu.getData(0)



    // 2) KERNEL 1: softmax + dLogits
    const probs = new Float32Array(vocab);
    const dLogits = new Float32Array(vocab);

    gpu.setData([
        { binding: 0, type: 'read-only-storage', input: logits, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
        { binding: 1, type: 'storage', input: probs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST },
        { binding: 2, type: 'storage', input: dLogits, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST },
        { binding: 3, type: 'uniform', input: new Uint32Array([target]), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    ])
    
    const wgsl_softmax_dlogits = fs.readFileSync('./shaders/kernel1.glsl').toString()
    gpu.runShader(wgsl_softmax_dlogits, {
        workgroupsX: Math.ceil(vocab / 256),
        workgroupsY: 1,
        workgroupsZ: 1
    })

    const dLogitsGPU = await gpu.getData(2)   // binding 2 = dLogits



    // 3) KERNEL 2: dW_out update
    gpu.setData([
        { binding: 0, type: 'read-only-storage', input: dLogitsGPU, usage: GPUBufferUsage.STORAGE },
        { binding: 1, type: 'read-only-storage', input: hLast, usage: GPUBufferUsage.STORAGE },
        { binding: 2, type: 'storage', input: Wout, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST },
        { binding: 3, type: 'uniform', input: new Uint32Array([vocab, d_model]), usage: GPUBufferUsage.UNIFORM },
        { binding: 4, type: 'uniform', input: new Float32Array([lr]), usage: GPUBufferUsage.UNIFORM },
    ])

    const wgsl_dWout = fs.readFileSync('./shaders/kernel2.glsl').toString()
    gpu.runShader(wgsl_dWout, {
        workgroupsX: Math.ceil(vocab / 16),
        workgroupsY: Math.ceil(d_model / 16),
        workgroupsZ: 1
    })

    await gpu.getData(2) // Wout zaktualizowany (binding 2)



    // 4) KERNEL 3: dH_last + update embeddingu ostatniego tokena
    const dHlast = new Float32Array(d_model)

    gpu.setData([
        { binding: 0, type: 'read-only-storage', input: dLogitsGPU, usage: GPUBufferUsage.STORAGE },
        { binding: 1, type: 'read-only-storage', input: Wout, usage: GPUBufferUsage.STORAGE },
        { binding: 2, type: 'storage', input: dHlast, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC },
        { binding: 3, type: 'storage', input: W_in_last, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST },
        { binding: 4, type: 'uniform', input: new Uint32Array([vocab, d_model]), usage: GPUBufferUsage.UNIFORM },
        { binding: 5, type: 'uniform', input: new Float32Array([lr]), usage: GPUBufferUsage.UNIFORM },
    ])

    const wgsl_dHlast_updateEmb = fs.readFileSync('./shaders/kernel3.glsl').toString()
    gpu.runShader(wgsl_dHlast_updateEmb, {
        workgroupsX: Math.ceil(d_model / 256),
        workgroupsY: 1,
        workgroupsZ: 1
    })

    const dHlastGPU = await gpu.getData(2)          // gradient po hLast
    const W_in_last_updated = await gpu.getData(3)  // zaktualizowany embedding


console.log('GPU backprop LM-head + embedding update wykonany.')
}