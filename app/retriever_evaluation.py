# """
# NeuroAssist Multi-ID Retriever Evaluation Script

# Purpose:
# Evaluate Retriever Accuracy supporting multiple valid target chunk IDs.

# This script:
# 1. Uses ONLY the retriever pipeline
# 2. Does NOT use the LLM
# 3. Loads:
#    retriever_{tumor_type}_test.json
# 4. Compares:
#    Retrieved Top Chunk IDs
#    VS
#    List of Valid Target Chunk IDs (Any intersection = Correct)
# 5. Computes:
#    Retriever Accuracy

# Run:
# python retriever_evaluation.py
# """

# import json
# import traceback
# import asyncio
# from typing import List
# from dataclasses import dataclass

# # ============================================================
# # IMPORTS
# # ============================================================

# from app.tools.prompt import router_prompt
# from app.tools.pipeline import BotPipeline
# from app.tools.vectordatabase import pineretriever
# from app.tools.agents import chat_model
# from app.core.config import settings

# print("--------------------------------------------------")
# print("All Imports Successful")
# print("--------------------------------------------------")


# # ============================================================
# # GLOBAL INITIALIZATION
# # SAME AS main.py
# # ============================================================

# print("\n--------------------------------------------------")
# print("Initializing Retriever Pipeline")
# print("--------------------------------------------------")

# serp_api_key = settings.SERP_API_KEY
# llm = chat_model

# pipeline = BotPipeline(
#     llm=llm,
#     router_prompt=router_prompt,
#     retriever=pineretriever,
#     serp_api_key=serp_api_key,
#     k=7,
#     alpha=0.4,
#     top_n=4
# )

# print("Retriever Pipeline Initialized Successfully")
# print("--------------------------------------------------")


# # ============================================================
# # RESULT DATA CLASS
# # ============================================================

# @dataclass
# class RetrieverEvaluationResult:
#     question: str
#     target_chunk_ids: List[int]
#     retrieved_chunk_ids: List[int]
#     correct: bool


# # ============================================================
# # EVALUATOR CLASS
# # ============================================================

# class RetrieverEvaluator:

#     def __init__(self, tumor_type: str):
#         self.tumor_type = tumor_type.strip().lower()
#         self.dataset_path = (
#             f"app/EvaluationData/EvaluationInputData/"
#             f"retriever_{self.tumor_type}_test.json"
#         )
#         self.test_data = self.load_test_data()
#         self.results: List[RetrieverEvaluationResult] = []

#         # Default Context
#         self.city = "pune"
#         self.state = "maharashtra"
#         self.country = "india"

#     # ========================================================
#     # LOAD DATASET
#     # ========================================================

#     def load_test_data(self):
#         try:
#             with open(self.dataset_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)

#                 print("\n--------------------------------------------------")
#                 print(f"Loaded {len(data)} Retriever Samples")
#                 print(f"Dataset : {self.dataset_path}")
#                 print("--------------------------------------------------")
#                 return data
#         except Exception as e:
#             print("\n--------------------------------------------------")
#             print("ERROR LOADING DATASET")
#             print(str(e))
#             print("--------------------------------------------------")
#             raise

#     # ========================================================
#     # CONTEXT SETTER
#     # ========================================================

#     def set_context(self, city: str, state: str, country: str):
#         self.city = city
#         self.state = state
#         self.country = country

#     # ========================================================
#     # GET TOP RETRIEVED CHUNK
#     # ========================================================

#     async def get_top_chunk_ids(self, question: str) -> List[int]:
#         try:
#             chat_history = []
#             final_context, retrieved_results, num_docs = await pipeline.run(
#                 query=question,
#                 chat_history=chat_history,
#                 tumor_type=self.tumor_type,
#                 city=self.city,
#                 state=self.state,
#                 country=self.country,
#             )

#             print("=" * 80)
#             print("QUESTION:", question)
#             print("NUM DOCS:", num_docs)
#             print("RETRIEVED RESULTS:", retrieved_results)
#             print("=" * 80)

#             if not retrieved_results:
#                 return []
            
#             retrieved_chunk_ids = []
#             for doc in retrieved_results:
#                 chunk_id = doc.metadata.get('chunk_id', -1)
#                 retrieved_chunk_ids.append(chunk_id)

#             return retrieved_chunk_ids
            
#         except Exception:
#             print("\n--------------------------------------------------")
#             print("RETRIEVER ERROR")
#             print("--------------------------------------------------")
#             traceback.print_exc()
#             return []

#     # ========================================================
#     # EVALUATE
#     # ========================================================

#     async def evaluate(self, batch_size: int = 10) -> float:
#         print("\n==================================================")
#         print("STARTING RETRIEVER EVALUATION")
#         print("==================================================")
#         print(f"TUMOR TYPE : {self.tumor_type}")
#         print(f"BATCH SIZE : {batch_size}")
#         print("==================================================")

#         selected_samples = self.test_data[:batch_size]
#         self.results = []
#         correct_predictions = 0

#         for idx, sample in enumerate(selected_samples, start=1):
#             question = sample["question"]
#             # Handle dataset variations safely (fallback to legacy singular key if needed)
#             target_chunk_ids = sample.get("target_chunk_ids", [sample.get("target_chunk_id")])

#             print("\n--------------------------------------------------")
#             print(f"SAMPLE {idx}/{batch_size}")
#             print("--------------------------------------------------")
#             print(f"\nQUESTION :\n{question}")

#             retrieved_chunk_ids = await self.get_top_chunk_ids(question)

#             # Check for any valid intersection between retrieved targets and candidate IDs
#             is_correct = any(tid in retrieved_chunk_ids for tid in target_chunk_ids)

#             if is_correct:
#                 correct_predictions += 1

#             result = RetrieverEvaluationResult(
#                 question=question,
#                 target_chunk_ids=target_chunk_ids,
#                 retrieved_chunk_ids=retrieved_chunk_ids,
#                 correct=is_correct
#             )
#             self.results.append(result)

#             print("\nRESULT")
#             print("--------------------------------------------------")
#             print(f"TARGET CHUNK IDS    : {target_chunk_ids}")
#             print(f"RETRIEVED CHUNK IDS : {retrieved_chunk_ids}")
#             print(f"CORRECT             : {is_correct}")
#             print("--------------------------------------------------")

#         # ------------------------------------------------------
#         # FINAL ACCURACY
#         # ------------------------------------------------------
#         accuracy = (correct_predictions / batch_size) * 100

#         print("\n==================================================")
#         print("FINAL RETRIEVER ACCURACY")
#         print("==================================================")
#         print(f"TOTAL SAMPLES     : {batch_size}")
#         print(f"CORRECT MATCHES   : {correct_predictions}")
#         print(f"ACCURACY          : {accuracy:.2f}%")
#         print("==================================================")

#         return accuracy

#     # ========================================================
#     # EXPORT RESULTS
#     # ========================================================

#     def export_results(self):
#         output_path = (
#             f"app/EvaluationData/EvaluationOutputData/"
#             f"{self.tumor_type}_retriever_results.json"
#         )

#         total = len(self.results)
#         correct = sum(1 for r in self.results if r.correct)
#         accuracy = (correct / total) * 100 if total > 0 else 0

#         export_data = {
#             "metadata": {
#                 "tumor_type": self.tumor_type,
#                 "city": self.city,
#                 "state": self.state,
#                 "country": self.country,
#             },
#             "summary": {
#                 "total_samples": total,
#                 "correct_matches": correct,
#                 "accuracy": accuracy
#             },
#             "results": [
#                 {
#                     "question": r.question,
#                     "target_chunk_ids": r.target_chunk_ids,
#                     "retrieved_chunk_ids": r.retrieved_chunk_ids,
#                     "correct": r.correct
#                 }
#                 for r in self.results
#             ]
#         }

#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(export_data, f, indent=2, ensure_ascii=False)

#         print("\n--------------------------------------------------")
#         print(f"Results Exported : {output_path}")
#         print("--------------------------------------------------")


# # ============================================================
# # USER INPUT
# # ============================================================

# def get_user_input():
#     print("\n==================================================")
#     print("RETRIEVER EVALUATION SETUP")
#     print("==================================================")

#     tumor_type = (input("\nEnter Tumor Type : ") or "glioma").lower()
#     batch_size = int(input("Enter Batch Size : ") or "10")
#     city = (input("Enter City : ") or "pune").lower()
#     state = (input("Enter State : ") or "maharashtra").lower()
#     country = (input("Enter Country : ") or "india").lower()

#     return tumor_type, batch_size, city, state, country


# # ============================================================
# # MAIN
# # ============================================================

# async def main():
#     (
#         tumor_type,
#         batch_size,
#         city,
#         state,
#         country
#     ) = get_user_input()

#     evaluator = RetrieverEvaluator(tumor_type=tumor_type)
#     evaluator.set_context(city=city, state=state, country=country)

#     await evaluator.evaluate(batch_size=batch_size)
#     evaluator.export_results()

#     print("\n==================================================")
#     print("RETRIEVER EVALUATION COMPLETED")
#     print("==================================================")


# if __name__ == "__main__":
#     asyncio.run(main())
