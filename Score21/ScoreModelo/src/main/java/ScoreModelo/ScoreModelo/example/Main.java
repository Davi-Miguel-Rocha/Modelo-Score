package ScoreModelo.ScoreModelo.example;

import org.nd4j.linalg.dataset.DataSet; //Aqui estão tanto os dados de entrada quanto as saídas esperadas.
import org.deeplearning4j.nn.conf.NeuralNetConfiguration; //Aqui são os hiperparâmetros da rede
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File; // IMPORTANTE: Adicionado para salvar
import java.io.IOException;
import org.deeplearning4j.util.ModelSerializer; //IMPORTANTE: pra salvar o modelo
import java.io.FileOutputStream; // NOVO IMPORT
import java.io.ObjectOutputStream; // NOVO IMPORT

//PRA DAR RUN NO CÓDIGO, DIGITE ./gradlew run NO TERMINAL.

public class Main {

	// VARIÁVEIS GLOBAIS
	private static final String MODEL_FILE_PATH = "credit_score_model2.zip";
	private static final String NORMALIZER_FILE_PATH = "credit_score_normalizer2.bin";


	public static void main(String[] args) {

		int input_Layer = 8;

		int camada_oculta = 64;


		int camada_saida = 1;


		MultiLayerNetwork modelo = new MultiLayerNetwork(

				new NeuralNetConfiguration.Builder()

						.weightInit(WeightInit.XAVIER)
						.updater(new Adam(0.0005)) //Rate de aprendizado
						.list()
						.layer(new DenseLayer.Builder()
								.nIn(input_Layer)
								.nOut(128)
								.activation(Activation.RELU)// Função de ativação ReLU
								.build())
						// Segunda camada oculta
						.layer(new DenseLayer.Builder()
								.nIn(128)
								.nOut(64)
								.activation(Activation.LEAKYRELU) // Função de ativação Leaky Relu
								.build())
						// Terceira camada oculta
						.layer(new DenseLayer.Builder()
								.nIn(64)
								.nOut(32)
								.activation(Activation.RELU)
								.build())
						.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Erro quadrático médio
								.nIn(32)
								.nOut(camada_saida)
								.activation(Activation.IDENTITY) // Função de ativação Identity
								.build())
						.build()
		);

		modelo.init();
		System.out.println("A rede funcionou!"); //Verificando se a rede funcionou;

		DataSet ds = Datasetexemple.createDataset(); //Cria o Dataset

// Dividir 80% treino / 20% validação
		SplitTestAndTrain split = ds.splitTestAndTrain(0.8);
		DataSet treino = split.getTrain();
		DataSet validacao = split.getTest();

// Normalização automática
		NormalizerStandardize normalizer = new NormalizerStandardize();
		normalizer.fit(treino);           // calcula média e desvio do treino
		normalizer.transform(treino);     // aplica no treino
		normalizer.transform(validacao);  // aplica na validação

// Treinamento
		int epochs = 1000; // suficiente para poucos dados
		for (int i = 0; i < epochs; i++) {
			modelo.fit(treino);
			double lossTreino = modelo.score(treino);
			double lossValidacao = modelo.score(validacao);

			if (i % 10 == 0) {
				System.out.println("Epoch " + i + ": Treino=" + lossTreino + " | Validação=" + lossValidacao);
			}
		}
		System.out.println("Treinamento concluído!");

		// Cliente 1: ALTO RISCO (Score Baixo esperado)
		double[] clienteAltoRisco = new double[]{6, 3, 1, 20, 5500, 0.3, 36, 0};
		INDArray inputAltoRisco = Nd4j.create(clienteAltoRisco).reshape(1, 8);
		normalizer.transform(inputAltoRisco);
		INDArray outputAltoRisco = modelo.output(inputAltoRisco);
		double rawScoreAltoRisco = outputAltoRisco.getDouble(0);
		double scoreAltoRisco = Math.max(300, Math.min(1000, rawScoreAltoRisco)); // Clipping

		// Cliente 2: BAIXO RISCO (Score Alto esperado)
		double[] clienteBaixoRisco = new double[]{1, 0, 0, 1, 9000, 0.05, 108, 0};
		INDArray inputBaixoRisco = Nd4j.create(clienteBaixoRisco).reshape(1, 8);
		normalizer.transform(inputBaixoRisco);
		INDArray outputBaixoRisco = modelo.output(inputBaixoRisco);
		double rawScoreBaixoRisco = outputBaixoRisco.getDouble(0);
		double scoreBaixoRisco = Math.max(300, Math.min(1000, rawScoreBaixoRisco)); // Clipping

		// Cliente 3: RISCO MÉDIO (Score esperado ~500)
		double[] clienteMedioRisco = new double[]{3, 1, 1, 5, 7000, 0.15, 60, 0};
		INDArray inputMedioRisco = Nd4j.create(clienteMedioRisco).reshape(1, 8);
		normalizer.transform(inputMedioRisco);
		INDArray outputMedioRisco = modelo.output(inputMedioRisco);
		double rawScoreMedioRisco = outputMedioRisco.getDouble(0);
		double scoreMedioRisco = Math.max(300, Math.min(1000, rawScoreMedioRisco)); // Clipping


		System.out.println("\n--- Validação do Comportamento do Modelo ---");
// OUTPUT CORRIGIDO (sem caracteres especiais para evitar erro de codificacao no console)
		System.out.println("\n--- Validacao do Comportamento do Modelo ---");
		System.out.println("1. Cliente de ALTO Risco (Score Bruto: " + (int)rawScoreAltoRisco + "): Score Final=" + (int)scoreAltoRisco);
		System.out.println("2. Cliente de BAIXO Risco (Score Bruto: " + (int)rawScoreBaixoRisco + "): Score Final=" + (int)scoreBaixoRisco);
		System.out.println("3. Cliente de MEDIO Risco (Score Bruto: " + (int)rawScoreMedioRisco + "): Score Final=" + (int)scoreMedioRisco);

		// --- SALVAR O MODELO E O NORMALIZADOR (O bloco try-catch corrigido) ---
		try {
			// 1. Salvar o Modelo Treinado
			File locationModel = new File(MODEL_FILE_PATH);
			boolean saveUpdater = true;
			ModelSerializer.writeModel(modelo, locationModel, saveUpdater);
			System.out.println("\n[SUCESSO] Modelo salvo em: " + MODEL_FILE_PATH);

			// 2. Salvar o Normalizador usando serialização Java (CORREÇÃO DO ERRO)
			File locationNormalizer = new File(NORMALIZER_FILE_PATH);
			try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(locationNormalizer))) {
				oos.writeObject(normalizer);
			}
			System.out.println("[SUCESSO] Normalizador salvo em: " + NORMALIZER_FILE_PATH);

		} catch (IOException e) {
			System.err.println("\n[ERRO FATAL] Falha ao salvar os arquivos de produção. Verifique permissões de arquivo.");
			e.printStackTrace();
		}


	}
}