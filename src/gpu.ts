import fs from 'fs'

let device:GPUDevice

const shaderCodes:any = {}

export class GPU {
    private device: GPUDevice

    private bindGroupLayout?: GPUBindGroupLayout
    private uniforms?: GPUBindGroupEntry[]
    //private configDataSet: any[] = []

    private shaderModule:GPUShaderModule

    private buffers:any = []

    private pipeline?:GPUComputePipeline
    private bindGroup?:GPUBindGroup 

    constructor(shaderFile:string) {
        this.device = device

        if(shaderCodes[shaderFile]==undefined){
            shaderCodes[shaderFile] = fs.readFileSync('./shaders/'+shaderFile).toString()
        }
        this.shaderModule = this.device.createShaderModule({ code: shaderCodes[shaderFile] })
    }

    setData(){
        if(!this.bindGroupLayout){
            const entries: GPUBindGroupLayoutEntry[] = []
            const resource: GPUBindGroupEntry[] = []

            this.buffers.map((buffer:any,i:number)=>{
                entries.push({
                    binding: i,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: buffer.type },
                })

                resource.push({
                    binding: i,
                    resource: { buffer:buffer.buffer },
                })
            })

            this.bindGroupLayout = this.device.createBindGroupLayout({
                entries,
            })

            this.uniforms = resource
        }
    }

    createBuffer(type: string, size: number, usage: GPUBufferUsageFlags) {
        // ZAWSZE dodajemy COPY_DST, bo używamy writeBuffer
        const buffer = this.device.createBuffer({
            size: size*4,
            usage,
        })
        this.buffers.push({
            type,
            buffer,
            usage,
        })
        return buffer
    }

    writeBuffer(buffer:GPUBuffer,data:GPUAllowSharedBufferSource){
        this.device.queue.writeBuffer(buffer, 0, data)
    }

    createPipeline(){
        this.pipeline = this.device.createComputePipeline({
            layout: this.bindGroupLayout
                ? this.device.createPipelineLayout({
                    bindGroupLayouts: [this.bindGroupLayout],
                })
                : "auto",
            compute: { module: this.shaderModule, entryPoint: "main" },
        })
        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout ?? this.pipeline.getBindGroupLayout(0),
            entries: this.uniforms!,
        })
    }

    runShader(workgroups: { x:number,y:number,z:number }) {
        this.setData()
        if(!this.pipeline){
            this.createPipeline()
        }

        const encoder = this.device.createCommandEncoder()
        const pass = encoder.beginComputePass()

        pass.setPipeline(this.pipeline!)
        pass.setBindGroup(0, this.bindGroup)

        pass.dispatchWorkgroups(
            workgroups.x,
            workgroups.y,
            workgroups.z,
        )

        pass.end()
        this.device.queue.submit([encoder.finish()])
    }

    async getData(buffer:GPUBuffer, length:number) {
        //const cfg = this.buffers[binding]

        //const buffer: GPUBuffer = cfg.buffer
        const byteLength = length * 4// = cfg.input.byteLength

        const readBuffer = this.device.createBuffer({
            size: byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })

        const encoder = this.device.createCommandEncoder()
        encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteLength)
        this.device.queue.submit([encoder.finish()])

        await this.device.queue.onSubmittedWorkDone()
        await readBuffer.mapAsync(GPUMapMode.READ)

        const result = new Float32Array(readBuffer.getMappedRange())

        return result
    }
}

export default GPU

export function setDevice(_device:GPUDevice){
    device = _device
}