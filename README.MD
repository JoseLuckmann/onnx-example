# ONNX Inference Examples

Este repositório contém exemplos de inferência de modelos ONNX em diferentes linguagens de programação e ambientes, incluindo Python, Java, C#, e JavaScript (tanto Node.js quanto Web). 

## O que é ONNX?

ONNX (Open Neural Network Exchange) é um formato aberto construído para representar modelos de aprendizado de máquina. ONNX define um conjunto comum de operadores e um formato de arquivo padrão para representar modelos de IA. ONNX foi criado para permitir a interoperabilidade entre diferentes ferramentas e frameworks de aprendizado de máquina, facilitando o processo de transferência de modelos entre diferentes ambientes e otimizando a execução em diferentes plataformas de hardware.

### Como o ONNX Funciona?

ONNX permite que os desenvolvedores treinem um modelo em um framework específico (por exemplo, PyTorch, TensorFlow, Scikit-learn) e, em seguida, exportem esse modelo para o formato ONNX. Este modelo ONNX pode então ser importado e executado em diferentes ambientes que suportam ONNX, como mecanismos de inferência, frameworks de aprendizado de máquina e plataformas de hardware acelerado.

### Capacidades do ONNX

- **Interoperabilidade**: Permite que modelos treinados em diferentes frameworks sejam compartilhados e utilizados em várias plataformas de inferência.
- **Suporte a Vários Frameworks**: ONNX é suportado por frameworks populares como PyTorch, TensorFlow, Scikit-learn, MXNet, entre outros.
- **Execução Otimizada**: ONNX Runtime fornece otimizações específicas de hardware para acelerar a inferência.
- **Expansibilidade**: Suporta a adição de novos operadores e extensões personalizadas.
- **Ferramentas de Conversão**: Ferramentas como `onnx-tf`, `tf2onnx`, `onnxmltools` facilitam a conversão de modelos entre diferentes formatos.

## Estrutura do Repositório

- `python/`: Exemplo de treinamento de ONNX usando Python.

- `java/`: Exemplo de inferência de ONNX usando Java.
- `csharp/`: Exemplo de inferência de ONNX usando C#.
- `javascript/`: Exemplo de inferência de ONNX usando JavaScript no Node.js.
- `javascript-web/`: Exemplo de inferência de ONNX usando JavaScript no navegador web.

## Como Usar os Exemplos

### Requisitos

- **Python**: Python 3.x, `onnx`, `onnxruntime`, `skl2onnx`, `scikit-learn`
- **Java**: JDK 11+, Maven
- **C#**: .NET SDK 6.0+
- **JavaScript**: Node.js, npm

### Configuração dos Exemplos
---
#### Java

Navegue até o diretório java/Inference:
```bash
cd java/Inference
```

Compile e execute o projeto Maven:
```bash
mvn compile
mvn exec:java -Dexec.mainClass="com.example.Inference"
```
---
#### C#

Navegue até o diretório csharp/OnnxInferenceCSharp:
```bash
cd csharp/
```

Compile e execute o projeto:
```bash
dotnet build
dotnet run
```
---
#### JavaScript (Node.js)

Navegue até o diretório javascript/node:
```bash
cd javascript/node
```

Instale as dependências e execute o script:
```bash
npm install
node index.js
```
---
#### JavaScript (Web)
Navegue até o diretório javascript/web:

```bash
cd javascript/web
```

Inicie um servidor HTTP:

```bash
http-server .
```

Abra o navegador e acesse http://localhost:8080.
---
## Contribuição
Contribuições são bem-vindas! Se você encontrar algum problema ou tiver sugestões de melhorias, por favor, abra uma issue ou envie um pull request.