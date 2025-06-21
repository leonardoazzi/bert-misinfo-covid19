#!/usr/bin/env python3
"""
Script para classificar notícias sobre COVID-19 usando Ollama API
Detecta fake news no dataset covidbr_labeled_cleaned.csv
"""

import os
import pandas as pd
import numpy as np
import requests
import json
from tqdm import tqdm
import time
from typing import List, Dict, Any
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OllamaFakeNewsClassifier:
    def __init__(
        self, base_url: str = "http://localhost:11434", model_name: str = "qwen3"
    ):
        """
        Inicializa o classificador de fake news com Ollama

        Args:
            base_url: URL base do servidor Ollama
            model_name: Nome do modelo a ser usado (ex: llama3.1, mistral, etc)
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_url = f"{self.base_url}/api/generate"

        # Verificar se o servidor Ollama está rodando
        self._check_ollama_server()

        # Verificar se o modelo está disponível
        self._check_model_availability()

        # Prompt para classificação de fake news
        self.classification_prompt = """

Você é um classificador de informações sobre COVID-19. Sua tarefa é analisar o conteúdo de mensagens de WhatsApp e determinar se elas são "Fake News" (1) ou "Verdadeiras" (0). Seja extremamente rigoroso para evitar classificar informações verdadeiras como "Fake News". Considere que uma mensagem é "Verdadeira" a menos que haja evidências claras e inquestionáveis de que ela contém informações enganosas, sensacionalistas, teorias da conspiração sem base científica, curas milagrosas não comprovadas ou dados falsificados relacionados à COVID-19.

**Instruções Específicas:**

1.  **FAKE NEWS (1):** Marque como "Fake News" apenas se a mensagem contiver um ou mais dos seguintes:
    * Alegações de curas ou tratamentos não comprovados cientificamente para COVID-19 (ex: "Chá de boldo cura COVID", "Ivermectina previne o vírus").
    * Teorias da conspiração sobre a origem do vírus, vacinas ou medidas de saúde (ex: "Vírus foi criado em laboratório para controle populacional", "Vacinas contêm chips de rastreamento").
    * Informações alarmistas ou sensacionalistas sem base em dados oficiais ou fontes confiáveis (ex: "Milhares de mortes escondidas", "Governo esconde a verdade sobre a pandemia").
    * Recomendações médicas perigosas ou contraindicações não endossadas por autoridades de saúde (ex: "Respirar vapor de eucalipto mata o vírus nos pulmões").
    * Dados estatísticos manipulados ou inventados sobre casos, mortes ou eficácia de tratamentos/vacinas.
    * Conteúdo que promove pânico ou medo de forma irracional e sem embasamento.
    * Citações falsas ou atribuídas incorretamente a especialistas ou organizações de saúde.

2.  **VERDADEIRA (0):** Marque como "Verdadeira" se a mensagem:
    * Contiver informações baseadas em dados de fontes oficiais (OMS, Ministério da Saúde, instituições de pesquisa renomadas, universidades).
    * Descrever sintomas, métodos de prevenção (uso de máscara, distanciamento social, lavagem das mãos), ou recomendações de vacinação de acordo com as diretrizes das autoridades de saúde.
    * Relatar notícias ou atualizações de fontes de jornalismo confiáveis e verificadas.
    * Expressar opiniões pessoais ou experiências que não alegam ser fatos científicos ou médicos universais (ex: "Senti dor de cabeça depois da vacina", "Minha vizinha se recuperou bem"). *Atenção: A menos que essa opinião promova desinformação ou alegações falsas.*
    * Compartilhar informações de conscientização ou campanhas de saúde pública.


**Formato da Saída:**

Apenas o número 0 ou 1.

**Exemplos (para Treinamento/Few-shot learning):**

**Mensagem:** "Beber água morna com limão a cada 3 horas mata o coronavírus na garganta, comprovado!"
**Classificação:** 1

**Mensagem:** "O Ministério da Saúde recomenda o uso de máscaras em ambientes fechados e a higienização das mãos com álcool em gel para prevenir a COVID-19."
**Classificação:** 0

**Mensagem:** "Um estudo recente da Universidade X sugere que a variante Y é mais contagiosa."
**Classificação:** 0

**Mensagem:** "Médico famoso revela: As vacinas de COVID alteram seu DNA e te transformam em um robô."
**Classificação:** 1

**Mensagem:** "Estou com febre e tosse, acho que pode ser COVID. Vou procurar um posto de saúde."
**Classificação:** 0

**Mensagem:** "Recebi a primeira dose da vacina e senti um pouco de dor no braço, mas já passou."
**Classificação:** 0

**Mensagem:** "Foto mostra hospitais vazios no auge da pandemia, tudo mentira da mídia!"
**Classificação:** 1

** Mensagem: ** "{text}" 
**Classificação:** /no_think
"""

    def _check_ollama_server(self):
        """Verifica se o servidor Ollama está rodando"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Servidor Ollama conectado com sucesso")
            else:
                raise ConnectionError(
                    f"Servidor Ollama retornou status {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Não foi possível conectar ao servidor Ollama em {self.base_url}: {e}"
            )

    def _check_model_availability(self):
        """Verifica se o modelo está disponível"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]

            if self.model_name not in available_models:
                logger.warning(
                    f"Modelo {self.model_name} não encontrado. Modelos disponíveis: {available_models}"
                )
                if available_models:
                    logger.info(
                        f"Tentando usar o primeiro modelo disponível: {available_models[0]}"
                    )
                    self.model_name = available_models[0]
                else:
                    raise ValueError("Nenhum modelo disponível no Ollama")
            else:
                logger.info(f"Usando modelo: {self.model_name}")

        except Exception as e:
            logger.warning(f"Não foi possível verificar modelos disponíveis: {e}")
            logger.info(f"Continuando com modelo: {self.model_name}")

    def classify_text(self, text: str, max_retries: int = 3) -> int:
        """
        Classifica um texto como fake news (1) ou informação verdadeira (0)

        Args:
            text: Texto para classificar
            max_retries: Número máximo de tentativas

        Returns:
            int: 0 para informação verdadeira, 1 para fake news
        """
        prompt = self.classification_prompt.format(
            text=text[:2000]
        )  # Limitar tamanho do texto

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,  # Baixa temperatura para respostas mais consistentes
                        "top_p": 0.8,
                        "max_tokens": 1,
                    },
                }

                response = requests.post(self.api_url, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["response"].strip()

                    # Tentar extrair 0 ou 1 da resposta
                    if "0" in prediction and "1" not in prediction:
                        return 0
                    elif "1" in prediction and "0" not in prediction:
                        return 1
                    elif prediction.startswith("0"):
                        return 0
                    elif prediction.startswith("1"):
                        return 1
                    else:
                        logger.warning(
                            f"Resposta ambígua: {prediction}. Tentativa {attempt + 1}"
                        )
                        if attempt == max_retries - 1:
                            # Se todas as tentativas falharam, classificar como não fake news (0)
                            return 0
                else:
                    logger.error(
                        f"Erro na API: {response.status_code} - {response.text}"
                    )

            except Exception as e:
                logger.error(f"Erro na tentativa {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return 0  # Default para não fake news se todas as tentativas falharam

    def classify_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        batch_size: int = 10,
        delay: float = 1.0,
    ) -> List[int]:
        """
        Classifica um dataset completo

        Args:
            df: DataFrame com os textos
            text_column: Nome da coluna com os textos
            batch_size: Número de textos a processar antes de uma pausa
            delay: Delay entre batches (segundos)

        Returns:
            List[int]: Lista com as predições
        """
        predictions = []

        logger.info(f"Iniciando classificação de {len(df)} textos usando Ollama")

        for i, text in enumerate(tqdm(df[text_column], desc="Classificando textos")):
            if pd.isna(text):
                predictions.append(0)
                continue

            prediction = self.classify_text(str(text))
            predictions.append(prediction)

            # Fazer pausa a cada batch_size textos
            if (i + 1) % batch_size == 0 and i < len(df) - 1:
                logger.info(
                    f"Processados {i + 1}/{len(df)} textos. Pausando por {delay}s..."
                )
                time.sleep(delay)

        logger.info("Classificação concluída!")
        return predictions

    def evaluate_predictions(
        self, y_true: List[int], y_pred: List[int]
    ) -> Dict[str, Any]:
        """
        Avalia as predições e calcula métricas

        Args:
            y_true: Labels verdadeiros
            y_pred: Predições do modelo

        Returns:
            Dict com as métricas
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "f1_score_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_score_micro": f1_score(y_true, y_pred, average="micro"),
            "precision_class_0": precision_score(y_true, y_pred, pos_label=0),
            "recall_class_0": recall_score(y_true, y_pred, pos_label=0),
            "f1_score_class_0": f1_score(y_true, y_pred, pos_label=0),
            "precision_class_1": precision_score(y_true, y_pred, pos_label=1),
            "recall_class_1": recall_score(y_true, y_pred, pos_label=1),
            "f1_score_class_1": f1_score(y_true, y_pred, pos_label=1),
        }

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Classificador de fake news usando Ollama"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/covidbr_labeled_cleaned.csv",
        help="Caminho para o arquivo CSV",
    )
    parser.add_argument(
        "--model", type=str, default="llama3.1", help="Nome do modelo Ollama"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:11434",
        help="URL base do servidor Ollama",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Número de amostras para testar (None para todo o dataset)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Tamanho do batch para processamento"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay entre batches em segundos"
    )

    args = parser.parse_args()

    try:
        # Carregar dados
        logger.info(f"Carregando dados de {args.data_path}")
        df = pd.read_csv(args.data_path)

        # Verificar colunas necessárias
        if "text" not in df.columns or "labels" not in df.columns:
            raise ValueError("O CSV deve conter as colunas 'text' e 'labels'")

        # Usar amostra se especificado
        if args.sample_size:
            df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
            logger.info(f"Usando amostra de {len(df)} textos")

        # Remover textos vazios ou nulos
        df = df.dropna(subset=["text", "labels"])
        logger.info(f"Dataset final: {len(df)} textos")

        # Inicializar classificador
        classifier = OllamaFakeNewsClassifier(
            base_url=args.base_url, model_name=args.model
        )

        # Classificar textos
        predictions = classifier.classify_dataset(
            df, batch_size=args.batch_size, delay=args.delay
        )

        # Avaliar resultados
        y_true = df["labels"].tolist()
        metrics = classifier.evaluate_predictions(y_true, predictions)

        # Exibir resultados
        print("\n" + "=" * 50)
        print("RESULTADOS DA CLASSIFICAÇÃO")
        print("=" * 50)
        print(f"Modelo usado: {args.model}")
        print(f"Total de textos: {len(df)}")
        print(f"Acurácia: {metrics['accuracy']:.4f}")
        print(f"Precisão: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_score_macro']:.4f}")
        print(f"F1-Score (Micro): {metrics['f1_score_micro']:.4f}")
        print("\nMétricas por classe:")
        print(
            f"Classe 0 (Verdadeiro) - Precisão: {metrics['precision_class_0']:.4f}, Recall: {metrics['recall_class_0']:.4f}, F1: {metrics['f1_score_class_0']:.4f}"
        )
        print(
            f"Classe 1 (Fake News) - Precisão: {metrics['precision_class_1']:.4f}, Recall: {metrics['recall_class_1']:.4f}, F1: {metrics['f1_score_class_1']:.4f}"
        )

        # Relatório detalhado
        print("\nRelatório de Classificação:")
        print(
            classification_report(
                y_true, predictions, target_names=["Verdadeiro", "Fake News"]
            )
        )

        # Matriz de confusão
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y_true, predictions)
        print(cm)

        # Salvar resultados
        results_df = df.copy()
        results_df["predicted_labels"] = predictions
        results_df["correct_prediction"] = (
            results_df["labels"] == results_df["predicted_labels"]
        )

        output_file = (
            f"ollama_{args.model.replace(':', '_')}_classification_results.csv"
        )
        results_df.to_csv(output_file, index=False)
        logger.info(f"Resultados salvos em {output_file}")

        # Salvar métricas
        metrics_file = f"ollama_{args.model.replace(':', '_')}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Métricas salvas em {metrics_file}")

        # Exibir alguns exemplos de classificação incorreta
        incorrect = results_df[results_df["correct_prediction"] == False]
        if len(incorrect) > 0:
            print(f"\nExemplos de classificação incorreta ({len(incorrect)} total):")
            for i, row in incorrect.head(3).iterrows():
                print(f"\nTexto: {row['text'][:200]}...")
                print(
                    f"Label real: {row['labels']}, Predição: {row['predicted_labels']}"
                )

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()
