package ScoreModelo.ScoreModelo.example;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Classe para gerar o dataset de exemplo.
 * Aumentado para 5000 exemplos para melhor aprendizado e generalização.
 * Features de entrada BRUTAS. Scores de saída na escala 300-1000.
 */
public class Datasetexemple {

    public static DataSet createDataset() {
        int totalExemplos = 5000; // AUMENTADO o número de exemplos para estabilizar o treinamento
        double[][] features = new double[totalExemplos][8];
        double[][] scores = new double[totalExemplos][1];

        final double MAX_CREDIT_SCORE = 1000.0;

        for (int i = 0; i < totalExemplos; i++) {

            // --- Gerando features BRUTAS e PLAUSÍVEIS ---
            double emprestimos = (int)(Math.random() * 15);
            double financiamentos = (int)(Math.random() * 7);
            double atrasos12meses = (int)(Math.random() * 12);
            double maiorAtraso = (int)(Math.random() * 60);
            double renda = 1000 + Math.random() * 14000;
            double comprometimento = Math.random() * 0.8;
            double idade = 18 + (int)(Math.random() * 50);
            double historicoRestricao = Math.random() < 0.2 ? 1.0 : 0.0;

            // --- Atribuindo features BRUTAS à entrada ---
            features[i] = new double[]{
                    emprestimos,
                    financiamentos,
                    atrasos12meses,
                    maiorAtraso,
                    renda,
                    comprometimento,
                    idade,
                    historicoRestricao
            };

            // --- Gerando o Score (300 a 1000) ---
            double scoreLogic =
                    (50 * (7 - emprestimos)) +
                            (100 * (5 - financiamentos)) +
                            (-50 * atrasos12meses) +
                            (-5 * maiorAtraso) +
                            (renda / 50.0) +
                            (-300 * comprometimento) +
                            (5 * idade) +
                            (-500 * historicoRestricao);

            // Ajusta o score para o range (ex: 300 a 1000)
            double scoreNormalized = Math.max(300, Math.min(MAX_CREDIT_SCORE, scoreLogic));

            scores[i] = new double[]{scoreNormalized};
        }

        INDArray input = Nd4j.create(features);
        INDArray output = Nd4j.create(scores);

        DataSet ds = new DataSet(input, output);
        ds.shuffle();

        return ds;
    }
}
