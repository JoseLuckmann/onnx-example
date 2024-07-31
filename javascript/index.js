const ort = require('onnxruntime-node');
const fs = require('fs');

async function run() {
    // Caminho para o modelo ONNX
    const modelPath = '../random_forest_iris.onnx';
    if (!fs.existsSync(modelPath)) {
        console.error(`Modelo ONNX não encontrado: ${modelPath}`);
        return;
    }

    // Carregar o modelo ONNX
    const session = await ort.InferenceSession.create(modelPath);

    // Criar dados de entrada
    const inputData = Float32Array.from([40.1, 30.5, 10.4, 30.2]);
    const tensor = new ort.Tensor('float32', inputData, [1, inputData.length]);

    const results = await session.run({ "float_input": tensor });
    // Imprimir os resultados
    console.log(`Previsão: ${results["label"].data}`);
    
}

run();

