let device

const shaderCodes = {}

export class GPU {
    device

    bindGroupLayout
    uniforms = []
    //private configDataSet: any[] = []

    shaderModule

    buffers = []

    pipeline
    bindGroup 

    encoder
    pass

    static async create(shaderFile) {
        const gpu = new GPU()
        gpu.device = device

        if(shaderCodes[shaderFile]==undefined){
            const promise = fetch('/'+shaderFile)
            const text = await promise.then(res=>res.text())
            shaderCodes[shaderFile] = text
        }
        gpu.shaderModule = gpu.device.createShaderModule({ code: shaderCodes[shaderFile] })
        return gpu
    }

    setData(){
        if(!this.bindGroupLayout){
            const entries = []
            const resource = []

            this.buffers.map((buffer,i)=>{
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

    createBuffer(dataset){
        const buffer = dataset.buffer??this.device.createBuffer({
            size: dataset.size*4,
            usage: dataset.usage,
        })
        this.buffers[dataset.binding] = {
            type: dataset.type,
            buffer,
            usage: dataset.usage,
        }
        return buffer
    }

    writeBuffer(buffer,data){
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
            entries: this.uniforms,
        })
    }

    begin(){
        this.setData()
        if(!this.pipeline){
            this.createPipeline()
        }

        this.encoder = this.device.createCommandEncoder()
        this.pass = this.encoder.beginComputePass()
    }

    end(){
        this.pass.end()
        this.device.queue.submit([this.encoder.finish()])
    }

    runShader(workgroups) {
        this.pass.setPipeline(this.pipeline)
        this.pass.setBindGroup(0, this.bindGroup)
        this.pass.dispatchWorkgroups(
            workgroups.x,
            workgroups.y,
            workgroups.z,
        )
    }

    async getData(buffer, length) {
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

export function setDevice(_device){
    device = _device
}