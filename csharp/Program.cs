using System;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxInferenceCSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Caminho para o modelo ONNX
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "../random_forest_iris.onnx");

            // Verificar se o modelo existe
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"O modelo ONNX não foi encontrado: {modelPath}");
                return;
            }

            // Carregar o modelo ONNX
            using var session = new InferenceSession(modelPath);

            // Criar dados de entrada
            var inputData = new float[] { 40.1f, 30.5f, 10.4f, 30.2f };
            var inputTensor = new DenseTensor<float>(inputData, new[] { 1, inputData.Length });

            // Criar o contêiner de entrada
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
            };

            // Fazer a previsão
            using var results = session.Run(inputs);
            var resultTensor = results.First().AsTensor<long>();
            var predictedClass = resultTensor.First();

            // Imprimir o resultado
            Console.WriteLine($"Previsão: {predictedClass}");
        }
    }
}
