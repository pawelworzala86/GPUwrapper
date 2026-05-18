import fs from 'fs'

const shaderCodes:any = {}

export class GPU {
    private device: GPUDevice

    private bindGroupLayout?: GPUBindGroupLayout
    private uniforms?: GPUBindGroupEntry[]
    private configDataSet: any[] = []

    private shaderModule:GPUShaderModule

    constructor(device: GPUDevice, shaderFile:string) {
        this.device = device

        if(shaderCodes[shaderFile]==undefined){
            shaderCodes[shaderFile] = fs.readFileSync('./shaders/'+shaderFile).toString()
        }
        this.shaderModule = this.device.createShaderModule({ code: shaderCodes[shaderFile] })
    }

    setData(configDataSet: any[]) {
        this.configDataSet = configDataSet

        const entries: GPUBindGroupLayoutEntry[] = []
        const resource: GPUBindGroupEntry[] = []

        configDataSet.map((config: any) => {
            // config.type MUSI być jednym z:
            // "storage" | "read-only-storage" | "uniform"
            entries.push({
                binding: config.binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: config.type },
            })

            const buffer = this.createBuffer(
                config.input,
                config.usage,
            )

            resource.push({
                binding: config.binding,
                resource: { buffer },
            })

            config.buffer = buffer
        })

        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries,
        })

        this.uniforms = resource
    }

    createBuffer(data: ArrayBufferView, usage: GPUBufferUsageFlags) {
        // ZAWSZE dodajemy COPY_DST, bo używamy writeBuffer
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: usage | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(buffer, 0, data)
        return buffer
    }

    runShader(workgroups: { workgroupsX: number, workgroupsY: number, workgroupsZ: number }) {
        const pipeline = this.device.createComputePipeline({
            layout: this.bindGroupLayout
                ? this.device.createPipelineLayout({
                    bindGroupLayouts: [this.bindGroupLayout],
                })
                : "auto",
            compute: { module: this.shaderModule, entryPoint: "main" },
        })

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout ?? pipeline.getBindGroupLayout(0),
            entries: this.uniforms!,
        })

        const encoder = this.device.createCommandEncoder()
        const pass = encoder.beginComputePass()

        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)

        pass.dispatchWorkgroups(
            workgroups.workgroupsX,
            workgroups.workgroupsY,
            workgroups.workgroupsZ,
        )

        pass.end()
        this.device.queue.submit([encoder.finish()])
    }

    async getData(binding: number) {
        const cfg = this.configDataSet.find((c: any) => c.binding === binding)
        const buffer: GPUBuffer = cfg.buffer
        const byteLength: number = cfg.input.byteLength

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
