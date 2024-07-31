package com.example;

import ai.onnxruntime.*;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;

public class Inference {
    public static void main(String[] args) throws OrtException {
      
        // Carregar o modelo ONNX
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession("../../random_forest_iris.onnx", new OrtSession.SessionOptions());

        // Criar dados de entrada
        float[] inputData = {40.1f, 30.5f, 10.4f, 30.2f}; 
        FloatBuffer inputBuffer = FloatBuffer.wrap(inputData);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer, new long[]{1, inputData.length});

        // Fazer a previsão
        OrtSession.Result result = session.run(Collections.singletonMap("float_input", inputTensor));
        long[] labels = (long[]) result.get(0).getValue();
        System.out.println("Previsão: " + labels[0]);
    }
}
