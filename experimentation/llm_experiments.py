from config import OPENAI_KEY, LLAMA_KEY

import pandas as pd
from llamaapi import LlamaAPI
from openai import OpenAI
import tqdm
import concurrent.futures
import random
import os

def request_batch(model_name, batch, client, examples = ""):
    """
    send requests and collects LLM responses

    batch: dataframe with "meeting_code", "piece_idx", and "transcript"
    examples: only for few-shot experiments

    returns a list of json objects with raw llm response
    request error: fill "Failed Request" in place of llm response
    """

    system_message = """
                    Imagine you are a meeting recorder who will be asked to segment the transcript of a city council meeting. Please segment based on the item discussed.
                    All segments are consecutive and non-overlapping, and the concatenation of them should consist the entire transcript. 
                    Therefore, for each segment, you need to copy the starting sentence exactly, token by token, to mark the segment divisions. 
                    Notice that the starting sentence of the first segment should be the starting sentence of the whole transcript.
                    If the starting sentence is not a unique sentence in the transcript, please also include the following sentence.
                    Please copy the string of sentence(s) exactly so that they can be searched within the transcript.
                    Please also write up to 2 sentences to summarize each segment/item. 
                    The core information for each segment is a motion/agenda being discussed and the decisions made about it. When it involves funding, the amount of funds is also important. Your principle of segmentation and summarization should center around such core information. 
                    Please respond in the format: number of segments: N, {"Segment1": {"starting sentence": "......", "summary": "......"}, "Segment2": {"starting sentence": "......", "summary": "......"}, ......, "SegmentN": {"starting sentence": "......", "summary": "......"}}
                    """

    batch_response = []

    for _,row in batch.iterrows():

        transcript = row['transcript']
        meeting_code = row['meeting_code']
        piece_idx = row['piece_idx']

        messages = [{"role": "system", "content": system_message}] + examples + [
            {"role": "user", "content": (
                f'Following the colon is the transcript, please do not include this prompt sentence before the colon: {transcript}'
            )}
        ]
        
        try:
            #make API call
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=700,
            )
            #observe system response
            system_response = response.choices[0].message.content.strip()
            print("\nSystem response: \n", system_response)

            batch_response.append({"meeting_code":meeting_code, "piece_idx": piece_idx, "raw_response":system_response})

        except Exception as e:
            print(f"Failed request for meeting {meeting_code} piece {piece_idx} model {model_name}: {e}")
            batch_response.append({"meeting_code":meeting_code, "piece_idx": piece_idx, "raw_response":"Failed Request"})

    return batch_response