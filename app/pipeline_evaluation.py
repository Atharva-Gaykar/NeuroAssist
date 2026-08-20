"""
NeuroAssist RAG Evaluation Script
Uses the SAME architecture as main.py
No FastAPI required

Evaluates:
- ROUGE-1
- ROUGE-L

Run:
python evaluate.py
"""

import json
import time
import traceback
from typing import List
from dataclasses import dataclass

from rouge_score import rouge_scorer

# ============================================================
# IMPORTS FROM MAIN.PY
# ============================================================

from app.tools.prompt import (
    medical_assistant_prompt,
    router_prompt,
)

from app.tools.pipeline import BotPipeline

from app.tools.vectordatabase import pineretriever

from app.tools.agents import chat_model

from app.core.config import settings

from langchain_core.messages import HumanMessage, AIMessage


print("--------------------------------------------------")
print("All imports successful")
print("--------------------------------------------------")


# ============================================================
# GLOBAL INITIALIZATION
# SAME AS main.py
# ============================================================

print("\n--------------------------------------------------")
print("Initializing Global Pipeline")
print("--------------------------------------------------")

serp_api_key = settings.SERP_API_KEY

llm = chat_model

print(f"LLM TYPE              : {type(llm)}")
print(f"PROMPT TYPE           : {type(medical_assistant_prompt)}")
print(f"ROUTER PROMPT TYPE    : {type(router_prompt)}")
print(f"RETRIEVER TYPE        : {type(pineretriever)}")

pipeline = BotPipeline(
    llm=llm,
    router_prompt=router_prompt,
    retriever=pineretriever,
    serp_api_key=serp_api_key,
    k=7,
    alpha=0.4,
    top_n=5
)

print("\n--------------------------------------------------")
print("Global Pipeline Initialized Successfully")
print("--------------------------------------------------")


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class EvaluationResult:

    question: str
    expected_answer: str
    generated_answer: str

    rouge1_precision: float
    rouge1_recall: float
    rouge1_f1: float

    rougeL_precision: float
    rougeL_recall: float
    rougeL_f1: float

    chunk_id: int

    sources_count: int
    execution_time: float


# ============================================================
# EVALUATOR CLASS
# ============================================================

class NeuroAssistEvaluator:

    def __init__(
        self,
        tumor_type: str
    ):

        self.tumor_type = tumor_type.lower()

        self.test_data_path = (
            f"app/EvaluationData/"
            f"{self.tumor_type}_test_questions.json"
        )

        self.test_data = self.load_test_data()

        self.rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rougeL'],
            use_stemmer=True
        )

        self.results: List[EvaluationResult] = []

        # Default Location Context
        self.city = "pune"
        self.state = "maharashtra"
        self.country = "india"

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    def load_test_data(self):

        try:

            with open(
                self.test_data_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

                print("\n--------------------------------------------------")
                print(f"Loaded {len(data)} Evaluation Questions")
                print(f"Dataset : {self.test_data_path}")
                print("--------------------------------------------------")

                return data

        except FileNotFoundError:

            print("\n--------------------------------------------------")
            print("ERROR : Evaluation Dataset Not Found")
            print(f"Missing File : {self.test_data_path}")
            print("--------------------------------------------------")

            raise

        except Exception as e:

            print("\n--------------------------------------------------")
            print("ERROR LOADING DATASET")
            print(str(e))
            print("--------------------------------------------------")

            raise

    # ========================================================
    # CONTEXT SETTER
    # ========================================================

    def set_context(
        self,
        city: str,
        state: str,
        country: str
    ):

        self.city = city
        self.state = state
        self.country = country

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question: str
    ):

        try:

            chat_history = []

            final_context, retrieved_results, num_docs = pipeline.run(
                query=question,
                chat_history=chat_history,
                tumor_type=self.tumor_type,
                city=self.city,
                state=self.state,
                country=self.country,
            )

            chain = medical_assistant_prompt | llm

            response = chain.invoke({

                "context": final_context,
                "question": question

            })

            if hasattr(response, "content"):
                bot_reply = response.content
            else:
                bot_reply = str(response)

            return bot_reply, num_docs

        except Exception as e:

            print("\n--------------------------------------------------")
            print("PIPELINE ERROR")
            print("--------------------------------------------------")

            traceback.print_exc()

            return (
                f"Pipeline generation failed: {str(e)}",
                0
            )

    # ========================================================
    # CALCULATE ROUGE METRICS
    # ========================================================

    def calculate_metrics(
        self,
        reference: str,
        generated: str
    ):

        scores = self.rouge.score(reference, generated)

        rouge1 = scores["rouge1"]
        rougeL = scores["rougeL"]

        return {

            "rouge1_precision": rouge1.precision,
            "rouge1_recall": rouge1.recall,
            "rouge1_f1": rouge1.fmeasure,

            "rougeL_precision": rougeL.precision,
            "rougeL_recall": rougeL.recall,
            "rougeL_f1": rougeL.fmeasure,
        }

    # ========================================================
    # MAIN EVALUATION
    # ========================================================

    def evaluate(
        self,
        num_questions: int = 5
    ):

        print("\n==================================================")
        print("STARTING NEUROASSIST RAG EVALUATION")
        print("==================================================")

        print(f"TUMOR TYPE    : {self.tumor_type}")
        print(f"CITY          : {self.city}")
        print(f"STATE         : {self.state}")
        print(f"COUNTRY       : {self.country}")
        print(f"QUESTIONS     : {num_questions}")

        print("==================================================")

        selected_questions = self.test_data[:num_questions]

        self.results = []

        for idx, sample in enumerate(selected_questions, start=1):

            question = sample["question"]
            expected_answer = sample["answer"]

            chunk_id = sample.get("chunk_id", idx)

            print("\n--------------------------------------------------")
            print(f"QUESTION {idx}/{num_questions}")
            print("--------------------------------------------------")

            print(f"\nQUESTION :\n{question}")

            start_time = time.time()

            generated_answer, sources_count = self.generate_answer(
                question
            )

            execution_time = time.time() - start_time

            metrics = self.calculate_metrics(
                expected_answer,
                generated_answer
            )

            result = EvaluationResult(

                question=question,

                expected_answer=expected_answer,

                generated_answer=generated_answer,

                rouge1_precision=metrics["rouge1_precision"],
                rouge1_recall=metrics["rouge1_recall"],
                rouge1_f1=metrics["rouge1_f1"],

                rougeL_precision=metrics["rougeL_precision"],
                rougeL_recall=metrics["rougeL_recall"],
                rougeL_f1=metrics["rougeL_f1"],

                chunk_id=chunk_id,

                sources_count=sources_count,

                execution_time=execution_time
            )

            self.results.append(result)

            print("\nEXPECTED ANSWER :")
            print(expected_answer[:300])

            print("\nGENERATED ANSWER :")
            print(generated_answer[:300])

            print("\nMETRICS")
            print("--------------------------------------------------")

            print(f"ROUGE-1 Precision : {result.rouge1_precision:.4f}")
            print(f"ROUGE-1 Recall    : {result.rouge1_recall:.4f}")
            print(f"ROUGE-1 F1        : {result.rouge1_f1:.4f}")

            print()

            print(f"ROUGE-L Precision : {result.rougeL_precision:.4f}")
            print(f"ROUGE-L Recall    : {result.rougeL_recall:.4f}")
            print(f"ROUGE-L F1        : {result.rougeL_f1:.4f}")

            print()

            print(f"SOURCES USED      : {sources_count}")
            print(f"EXECUTION TIME    : {execution_time:.2f} seconds")

            print("--------------------------------------------------")

        return self.results

    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(self):

        if not self.results:

            print("\nNo results available")
            return

        avg_rouge1 = sum(
            r.rouge1_f1 for r in self.results
        ) / len(self.results)

        avg_rougeL = sum(
            r.rougeL_f1 for r in self.results
        ) / len(self.results)

        avg_sources = sum(
            r.sources_count for r in self.results
        ) / len(self.results)

        avg_time = sum(
            r.execution_time for r in self.results
        ) / len(self.results)

        print("\n==================================================")
        print("FINAL EVALUATION SUMMARY")
        print("==================================================")

        print(f"QUESTIONS EVALUATED : {len(self.results)}")

        print()

        print(f"AVERAGE ROUGE-1 F1 : {avg_rouge1:.4f}")
        print(f"AVERAGE ROUGE-L F1 : {avg_rougeL:.4f}")

        print()

        print(f"AVERAGE SOURCES    : {avg_sources:.2f}")
        print(f"AVERAGE TIME       : {avg_time:.2f} seconds")

        print("==================================================")

    # ========================================================
    # EXPORT RESULTS
    # ========================================================

    def export_results(self):

        output_path = (
            f"{self.tumor_type}_evaluation_results.json"
        )

        export_data = {

            "metadata": {

                "tumor_type": self.tumor_type,
                "city": self.city,
                "state": self.state,
                "country": self.country,
            },

            "results": [

                {

                    "question": r.question,

                    "expected_answer": r.expected_answer,

                    "generated_answer": r.generated_answer,

                    "chunk_id": r.chunk_id,

                    "sources_count": r.sources_count,

                    "execution_time": r.execution_time,

                    "rouge1": {

                        "precision": r.rouge1_precision,
                        "recall": r.rouge1_recall,
                        "f1": r.rouge1_f1,
                    },

                    "rougeL": {

                        "precision": r.rougeL_precision,
                        "recall": r.rougeL_recall,
                        "f1": r.rougeL_f1,
                    }
                }

                for r in self.results
            ]
        }

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                export_data,
                f,
                indent=2,
                ensure_ascii=False
            )

        print("\n--------------------------------------------------")
        print(f"Results Exported : {output_path}")
        print("--------------------------------------------------")


# ============================================================
# USER INPUT
# ============================================================

def get_user_input():

    print("\n==================================================")
    print("NEUROASSIST EVALUATION SETUP")
    print("==================================================")

    num_questions = int(
        input("\nEnter Number Of Questions : ") or "5"
    )

    tumor_type = (
        input("Enter Tumor Type : ") or "glioma"
    ).lower()

    city = (
        input("Enter City : ") or "pune"
    ).lower()

    state = (
        input("Enter State : ") or "maharashtra"
    ).lower()

    country = (
        input("Enter Country : ") or "india"
    ).lower()

    return (
        num_questions,
        tumor_type,
        city,
        state,
        country
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        num_questions,
        tumor_type,
        city,
        state,
        country
    ) = get_user_input()

    evaluator = NeuroAssistEvaluator(
        tumor_type=tumor_type
    )

    evaluator.set_context(
        city=city,
        state=state,
        country=country
    )

    evaluator.evaluate(
        num_questions=num_questions
    )

    evaluator.print_summary()

    evaluator.export_results()

    print("\n==================================================")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("==================================================")


# if __name__ == "__main__":

#     main()