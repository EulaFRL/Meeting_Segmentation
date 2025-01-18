from config import OPENAI_KEY, LLAMA_KEY, GROUND_TRUTH_DIRS_DICT, CITY_IN_FORMAT_DICT
import response_parsing

import json
import pandas as pd
from llamaapi import LlamaAPI
from openai import OpenAI
import tqdm
import concurrent.futures
import random
import os
import re

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

def compose_sample_response(meeting_path):
    """extracts one piece from the meeting, compose sample LLM response and returns the transcript along with the response"""

    with open(meeting_path, "r") as f:
        meeting = json.load(meeting_path)
    
    sample_piece = random.select(meeting)   #selects one piece from the meeting
    n_segments = sample_piece["#items"]
    # response format: "number of segments: N, {"Segment1": {"starting sentence": "......", "summary": "......"}, "Segment2": {"starting sentence": "......", "summary": "......"}, ......, "SegmentN": {"starting sentence": "......", "summary": "......"}}"
    response = f"number of segments: {n_segments}, {{"
    for i in range(n_segments):
        segment = {
            "starting sentence": response_parsing.transcript_2_sentences(sample_piece["ground-truth segments"][i])[0],
            "summary": sample_piece["summaries"][i]
        }
        response += f'"Segment{i}": {json.dumps(segment)}'  # Using json.dumps to ensure proper string escaping
        if i < n_segments - 1:
            response += ", "
        else:
            response += "}"

    return (sample_piece["transcript"], response)
    

def generate_examples(training_paths, num_examples):
    """
    takes a list of filepaths that contains all training meetings

    generates a given number of examples for few-shot experiments
    
    train_in_dirs: list of directories to the folders containing ground-truth instances
    """

    # all_meeting_filename_list = [] #list of all json files that contains ground truth meeting instances
    # for in_dir in train_in_dirs: #multiple directories because each city is stored in different folders
    #     all_meeting_filename_list.extend(os.listdir(in_dir))
    # if ".DS_Store" in all_meeting_filename_list:
    #     all_meeting_filename_list = [i for i in all_meeting_filename_list if i != ".DS_Store"]

    sampled_meeting_filepath_list = training_paths.sample(n=num_examples, random_state=42)
    sampled_instances = []
    for meeting_path in sampled_meeting_filepath_list:
        sampled_instances.append(compose_sample_response(meeting_path)) #returns (transcript,response)


    #empty list to store examples
    examples = []

    for transcript, response in sampled_instances:
        #add user and assistant examples for this score
        #assistant role displays the correct format of response, helps in parsing
        #identifiers are used as index here, only matched both identifiers can map response
        #keywords are displayed because model might not be able to recognised un-lemmatised keywords
        examples.append(
            {"role": "user", "content": (
                f'Following the colon is the transcript, please do not include this prompt sentence before the colon: {transcript}'
            )}
        )
        examples.append(
            {"role": "assistant", "content": ({response})}
        )

    return examples

def city_requests(model_name, client, city, city_filepaths, all_meeting_paths, n_threads, batch_size, few_shot=False):
    """multi-thread sending requests in batch for one model and one city council"""

    output_path = f"output_{model_name}_{city}.json"

    with concurrent.futures.ThreadPoolExecutor(n_threads) as executor:
        #to store future, one rating = one object = one future
        futures = []
        #divides instances into batches of n-threads
        #each batch contains 15 user queries
        for start in range(0, len(city_filepaths), n_threads):
            #batch: dataframe with "meeting_code", "piece_idx", and "transcript"
            batch = pd.DataFrame(columns=["meeting_code", "piece_idx", "transcript"])
            batch_paths = city_filepaths.iloc[start:start + batch_size]
            #a list to store model prediction
            batch_responses = []
            for meeting_path in batch_paths:
                meeting = json.load(meeting_path)
                for i in range(len(meeting)):
                    piece = meeting[i]
                    batch.loc[len(batch)] = [re.search(r'_(\d+)_', meeting_path).group(1), i, piece["transcript"]]
            training_paths = [path for path in all_meeting_paths if not path in batch_paths] #avoid overlapping with test examples in the batch
            examples = "" #zero shot
            if few_shot:
                examples = generate_examples(training_paths, num_examples)
            #submit process batch function
            futures.p(executor.submit(request_batch, model_name, batch, client, examples))

        #progress bar
        #saves each completed user query
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {model_name}"):
            try:
                result = future.result() #retrieving result
                batch_responses.extend(result) #save each result to results list
            except Exception as e:
                print(f"Error in processing batch: {e}")
        #results dataframe
        response_parsing.response_parsing(batch_responses, GROUND_TRUTH_DIRS_DICT[city], output_path, CITY_IN_FORMAT_DICT[city])
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    #TODO
    client = OpenAI(api_key=OPENAI_KEY)
    model_name = 'gpt-4o-mini'
    num_examples = 3
    n_threads = 15
    batch_size = 10

    all_meeting_paths = []
    city_filepath_dict = dict()
    #reading input files
    for city, in_dir in GROUND_TRUTH_DIRS_DICT.items():
        filepaths = os.listdir(in_dir)
        filepaths = [os.path.join(in_dir,name) for name in filepaths if not name.startswith(".")]
        city_filepath_dict[city] = filepaths
        all_meeting_paths.extend(filepaths)

    for city, city_filepaths in city_filepath_dict.items():
        city_requests(model_name, client, city, city_filepaths, all_meeting_paths, n_threads, batch_size, few_shot=False)