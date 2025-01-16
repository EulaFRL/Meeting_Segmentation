import os
import json
import re

def response_parsing(batch_responses, in_dir, output_path, in_format):
    """
    parses a batch of llm responses
    , reads from corresponsing ground-truth file
    , and extracts relevant information for evaluation, stores to evaluation file for the city council

    in_format: input filename format with a placeholder "meeting_code"
    """

    with open(output_path, "a") as out_file:
        for response in batch_responses:
            meeting_code = response["meeting_code"]
            piece_idx = response["piece_idx"]
            input_path = os.path.join(in_dir, in_format.replace("meeting_code", meeting_code))

            if not os.path.exists(input_path):
                print(f"Ground truth file for meeting code {meeting_code} not found. Skipping.")
                continue

            with open(input_path, "r") as gt_file:
                ground_truth = json.load(gt_file)[piece_idx]

            try:
                output = process_batch_response(response, ground_truth, meeting_code, piece_idx)
            except Exception as e:
                print(f"Error parsing meeting {meeting_code} piece {piece_idx}: {e}")
                output = {"meeting_code": meeting_code, "piece_idx": piece_idx, "error": "Format Error"}
            
            out_file.write(json.dumps(output) + "\n")

def process_batch_response(response, ground_truth, meeting_code, piece_idx):
    """Process a single batch response."""
    transcript = ground_truth['transcript']
    transcript_ls = transcript_2_sentences(transcript)

    transcript_dict = {}
    idx = -1
    for sentence_index, sentence in enumerate(transcript_ls):
        cur_pair = find_overlapping_sentence(sentence.strip(), transcript, idx)
        if cur_pair:
            transcript_dict[cur_pair] = sentence_index
            idx = cur_pair[1]

    gt_segments = ground_truth['ground-truth segments']
    gt_starts = []
    idx = -1
    for segment in gt_segments:
        cur_pair = find_overlapping_sentence(segment.strip(), transcript, idx)
        if cur_pair:
            gt_starts.append(cur_pair)
            idx = cur_pair[1]

    gt_starts_idx = [transcript_dict[start] for start in gt_starts if start in transcript_dict]
    gt_starts_idx.append(len(transcript_ls))

    raw_response = response['raw_response']
    if raw_response == "Failed Request":
        return {"meeting_code": meeting_code, "piece_idx": piece_idx, "error": "Failed Request"}
    
    parsed_response = extract_json(raw_response)

    if not parsed_response:
        raise ValueError(f"Invalid JSON format in response for meeting {meeting_code} piece {piece_idx}.")

    llm_summaries = []
    llm_starts_temp = []
    for segment in parsed_response.values():
        llm_summaries.append(segment['summary'])
        llm_starts_temp.append(segment['starting sentence'])

    llm_starts = []
    idx = -1
    for start_temp in llm_starts_temp:
        start = find_overlapping_sentence(start_temp.strip(), transcript, idx)
        if start:
            llm_starts.append(start)
            idx = start[1]
        else:
            return {"meeting_code": meeting_code, "piece_idx": piece_idx, "error": "Hallucination"}

    llm_starts_idx = [transcript_dict[start] for start in llm_starts if start in transcript_dict]
    llm_starts_idx.append(len(transcript_ls))

    output = {
        "meeting_code": meeting_code,
        "piece_idx": piece_idx,
        "segment IDs": ground_truth['item IDs'],
        "gt start indices": gt_starts_idx,
        "llm start indices": llm_starts_idx,
        "gt nSegs": ground_truth['#items'],
        "llm nSegs": len(parsed_response),
        "gt summaries": ground_truth['summaries'],
        "llm summaries": llm_summaries,
        # "transcript": transcript, #TODO
    }
    return output

def transcript_2_sentences(transcript):
    """Break a transcript into a list of sentences."""
    sentence_splitters = r'(?<=[.!?])\s+(?=[A-Za-z0-9])'
    splits = re.split(sentence_splitters, transcript)

    merged_splits = []
    for split in splits:
        if split.strip():
            merged_splits.append(split.strip())

    return merged_splits

def extract_json(input_string):
    """Extract JSON from a raw LLM response string."""
    input_string = input_string.strip().replace('\n', '')
    json_match = re.search(r'\{.*', input_string)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def find_overlapping_sentence(substring, long_string, cur_idx=None):
    """Find the range of a sentence in the transcript."""
    match = re.finditer(re.escape(substring), long_string)
    for m in match:
        start_index = m.start()
        end_index = m.end()
        if cur_idx and start_index < cur_idx:
            continue
        return start_index, end_index
    return None
