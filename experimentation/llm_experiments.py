import math
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
import time
import requests

def request_batch(model_name, batch, client, examples = [], max_retries=5, retry_delay=10):
    """
    Send requests and collect LLM responses with infinite retry mechanism for connection errors, with max_retries.
    For invalid_request_errors, no max_retries, keep removing examples until request succeeds.
    
    batch: dataframe with "meeting_code" and "transcript"
    examples: only for few-shot experiments
    
    returns a list of json objects with raw llm response.
    If the request fails, "Failed Request" is placed in the response.
    """
    system_message = """
                    Imagine you are a meeting recorder who will be asked to segment the transcript of a city council meeting. Please segment based on the agenda item discussed.
                    The standard for separating segments is the motions/agendas being discussed, the segment lengths can be long or short, while you should stick to distinguishing the issues being discussed.
                    Segments typically ends with a decision/conclusion/vote made on the topic, or simply moving on to the next issue.
                    All segments are consecutive and non-overlapping, and the concatenation of them should consist the entire transcript.
                    Therefore, for each segment, you only need to extract the starting subsrting to mark the segment divisions.
                    The substring should be a unique substring of the transcript, containing one to two sentences.
                    Please copy the substrings exactly so that they are exactly a substring of the transcript.
                    Whether there are grammatical errors or not, do not make any changes. Be extra mindful in copying exactly when it involves numbers.
                    Notice that the starting substring of the first segment should be the starting subsrting of the whole transcript.
                    At the beginning of your response, please identify the number of segments: N.
                    Please also make sure that your segments are in the order as they appear in the transcript.
                    Please respond in this format and replace Starting_Substring with the actual starting substring of each segment: number of segments: N, ["Starting_Substring1", "Starting_Substring2", ......, "Starting SubstringN"]
                    """
    
    batch_response = []

    for _, row in batch.iterrows():

        transcript = row['transcript']
        meeting_code = row['meeting_code']

        messages = [{"role": "system", "content": system_message}] + examples + [
            {"role": "user", "content": (
                f'Following the colon is the transcript, please do not include this prompt sentence before the colon: {transcript}'
            )}
        ]

        # print(f"Total number of tokens in request contents: {sum(len(message['content'].split()) for message in messages)}")
        message_len = [len(message['content'].split()) for message in messages]
        while len(str(messages).split()) >= 80000:
            messages.remove(messages[1])
            messages.remove(messages[2]) # remove one example

        print(f"Each message's length: {str(message_len)}")
        print(f"Total request length: {len(str(messages).split())} tokens")

        retry_attempts = 0

        while retry_attempts < max_retries:
            try:
                # Attempt API call
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1000,
                )
                
                # Observe system response
                system_response = response.choices[0].message.content.strip()
                print("\nSystem response: \n", system_response)

                # Store the successful response
                batch_response.append({
                    "meeting_code": meeting_code,
                    "raw_response": system_response
                })
                break  # Exit the retry loop on success

            except requests.exceptions.ConnectionError as e:  # Catch connection errors specifically
                retry_attempts += 1
                print(f"Connection error for meeting {meeting_code} model {model_name}: {e}")

                if retry_attempts < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    print(f"Max retries reached for meeting {meeting_code}. Marking as failed.")
                    batch_response.append({
                        "meeting_code": meeting_code,
                        "raw_response": "Failed Request"
                    })
                continue  #  retry on connection errors
            
            except Exception as e:  # For other types of errors (e.g., invalid input, rate limits)
                # if response.status_code == 400:
                #     error_data = response.json()
                #     if 'error' in error_data:
                #         error_message = error_data['error'].get('message', '')
                # if 'context_length_exceeded' in e:  # exceed context length
                if re.search('maximum context length',str(e)): # exceed context length
                    if len(messages) > 5: 
                        messages.remove(messages[1])
                        messages.remove(messages[2])
                        print(f"Retry request length: {len(str(messages).split())} tokens")
                        continue
                print(f"Failed request for meeting {meeting_code} model {model_name}: {e}")
                batch_response.append({
                    "meeting_code": meeting_code,
                    "raw_response": "Failed Request"
                })
                break  # Exit the retry loop for other errors

    return batch_response

def compose_sample_response(meeting_path, max_tokens):
    """
    extracts one piece from the meeting, compose sample LLM response and returns the transcript along with the response
    returns (transcript,response) for a piece with token length <= max_tokens or None
    """

    with open(meeting_path, "r") as f:
        meeting = json.load(f)
    
    # sample_piece = random.choice(meeting)   #selects one piece from the meeting
    sample_piece = None
    for piece in meeting:
        if len(piece['transcript'].split()) <= max_tokens:
            sample_piece = piece
            break
    
    if not sample_piece:
        return None

    n_segments = sample_piece["#items"]
    # (obsolete) response format: "number of segments: N, {"Segment1": {"starting sentence": "......", "summary": "......"}, "Segment2": {"starting sentence": "......", "summary": "......"}, ......, "SegmentN": {"starting sentence": "......", "summary": "......"}}"
    # response format: number of segments: N, ["Starting_Substring1", "Starting_Substring2", ......, "Starting SubstringN"]
    response = f"number of segments: {n_segments}, ["
    for i in range(n_segments):
        response += f'"{response_parsing.transcript_2_sentences(sample_piece["ground-truth segments"][i])[0]}"'
        # obsolete:
        # segment = {
        #     "starting sentence": response_parsing.transcript_2_sentences(sample_piece["ground-truth segments"][i])[0],
        #     "summary": sample_piece["summaries"][i]
        # }
        # response += f'"Segment{i}": {json.dumps(segment)}'  # Using json.dumps to ensure proper string escaping
        if i < n_segments - 1:
            response += ", "
        else:
            response += "]"

    return (sample_piece["transcript"], response)
    

def generate_examples(training_paths, num_examples, max_tokens):
    """
    takes a list of filepaths that contains all training meetings

    generates a given number of examples for few-shot experiments
    
    train_in_dirs: list of directories to the folders containing ground-truth instances
    max_tokens: maximum token length for a sample piece's transcript
    """

    # all_meeting_filename_list = [] #list of all json files that contains ground truth meeting instances
    # for in_dir in train_in_dirs: #multiple directories because each city is stored in different folders
    #     all_meeting_filename_list.extend(os.listdir(in_dir))
    # if ".DS_Store" in all_meeting_filename_list:
    #     all_meeting_filename_list = [i for i in all_meeting_filename_list if i != ".DS_Store"]

    examples_count = 0
    #empty list to store examples
    examples = []
    while examples_count < num_examples:
        example_path = random.choice(training_paths)
        training_paths.remove(example_path)
        #returns (transcript,response) for a piece with transcript token length <= max_tokens or None
        example = compose_sample_response(example_path, max_tokens)
        if not example:
            continue
        examples.append(
            {"role": "user", "content": (
                f'Following the colon is the transcript, please do not include this prompt sentence before the colon: {example[0]}'
            )}
        )
        examples.append(
            {"role": "assistant", "content": example[1]}
        )
        examples_count += 1

    return examples

def city_requests(model_name, client, city, city_filepaths, all_meeting_paths, n_threads, batch_size, max_tokens, few_shot=False):
    """multi-thread sending requests in batch for one model and one city council"""

    output_path = f"output_{model_name}_{city}.json"

    with concurrent.futures.ThreadPoolExecutor(n_threads) as executor:
        #to store future, one rating = one object = one future
        futures = []
        #divides instances into batches of n-threads
        #each batch contains 15 user queries
        for start in range(0, len(city_filepaths), n_threads):
            #batch: dataframe with "meeting_code", and "transcript"
            batch = pd.DataFrame(columns=["meeting_code", "transcript"])
            batch_paths = city_filepaths[start:start + batch_size]
            #a list to store model prediction
            # batch_responses = []
            for meeting_path in batch_paths:
                with open(meeting_path) as f:
                    meeting = json.load(f)
                meeting_transcript = ""
                for i in range(len(meeting)):
                    meeting_transcript += meeting[i]['transcript']
                batch.loc[len(batch)] = [re.search(r'_(\d+)_', meeting_path).group(1), meeting_transcript]
            training_paths = [path for path in all_meeting_paths if not path in batch_paths] #avoid overlapping with test examples in the batch
            examples = [] #zero shot
            if few_shot:
                examples = generate_examples(training_paths, num_examples, max_tokens)
            #submit process batch function
            futures.append(executor.submit(request_batch, model_name, batch, client, examples))

        #progress bar
        #saves each completed user query
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {model_name}"):
            # try:
                batch_responses = future.result() #retrieving result
                response_parsing.response_parsing(batch_responses, GROUND_TRUTH_DIRS_DICT[city], output_path, CITY_IN_FORMAT_DICT[city])
                print(f"Batch results saved to {output_path}")
            # except Exception as e:
            #     print(f"Error in processing batch: {e}")
        # #results dataframe
        # response_parsing.response_parsing(batch_responses, GROUND_TRUTH_DIRS_DICT[city], output_path, CITY_IN_FORMAT_DICT[city])
        print(f"All city results saved to {output_path}")


if __name__ == "__main__":
    """
    without examples: a lot of hallucinations and high number of segments
    with examples: a lot of hallucinations and relatively more stable number of segments
    """

    #TODO
    client = OpenAI(api_key=OPENAI_KEY)
    model_name = 'gpt-4o-mini'
    num_examples = 10 #TODO
    max_tokens = 5000 #maximum token limit on the length of sample transcipts
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
        city_requests(model_name, client, city, city_filepaths, all_meeting_paths, n_threads, batch_size, max_tokens, few_shot=True)