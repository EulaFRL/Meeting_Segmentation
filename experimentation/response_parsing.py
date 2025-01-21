import ast
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
            input_path = os.path.join(in_dir, in_format.replace("meeting_code", meeting_code))

            if not os.path.exists(input_path):
                print(f"Ground truth file for meeting code {meeting_code} not found. Skipping.")
                continue

            with open(input_path, "r") as gt_file:
                ground_truth = json.load(gt_file)

            try:
                output = process_batch_response(response, ground_truth, meeting_code)
            except Exception as e:
                print(f"Error parsing meeting {meeting_code}: {e}")
                output = {"meeting_code": meeting_code, "error": "Format Error"}
            
            out_file.write(json.dumps(output) + "\n")

def process_batch_response(response, ground_truth, meeting_code):
    """
    Process a single response.
    ground_truth is a list of pieces in the meeting
    """
    meeting_transcript = ""
    gt_nSegs = 0
    gt_segments = [] #segment transcripts
    meeting_segIDs = []
    for piece in ground_truth:
        meeting_transcript += piece['transcript']
        gt_nSegs += piece['#items']
        gt_segments.extend(re.sub(r'\s*SPEAKER_\d{2}:\s*', ' ', segment) for segment in piece['ground-truth segments'])  # remove all speaker labels for evaluation
        meeting_segIDs.extend(piece['item IDs'])
    # transcript without speaker labels is used in all evaluations, while the original is used only in prompting
    meeting_transcript_wo_speaker = re.sub(r'\s*SPEAKER_\d{2}:\s*', ' ', meeting_transcript)
    transcript_ls = transcript_2_sentences(meeting_transcript_wo_speaker)

    transcript_dict = dict()  # {(sentence_start_char_idx, end) : sentence_idx}
    idx = -1  # end of last sentence
    previous_end = None
    previous_pair = None
    for sentence_index, sentence in enumerate(transcript_ls):
        # first char index ~ last char index(inclusive)
        cur_pair = find_overlapping_sentence(sentence.strip(), meeting_transcript_wo_speaker, idx)
        if not cur_pair:
            print(f"sentence {sentence.strip()}")
        # If there is a gap between the current pair and the previous one, fill the gap
        if previous_end is not None and cur_pair[0] > previous_end + 1:
            # Add the missing range to the dictionary
            missing_pair = (previous_end + 1, cur_pair[0] - 1)
            transcript_dict[missing_pair] = transcript_dict[previous_pair] + 1  # Increment the value consecutively
        # Update the dictionary with the current pair
        transcript_dict[cur_pair] = transcript_dict.get(previous_pair, 0) + 1 if previous_pair else 0
        # Update the previous range and index
        previous_end = cur_pair[1]
        previous_pair = cur_pair
        idx = cur_pair[1]

    gt_starts = []
    # idx = -1
    for segment in gt_segments:
        cur_pair = find_overlapping_sentence(segment.strip(), meeting_transcript_wo_speaker)
        if not cur_pair:
            print(meeting_transcript_wo_speaker)
            print(segment)
        gt_starts.append(cur_pair)
        # idx = cur_pair[1]\
    temp_transcript_dict = {char_idx[0]:sent_idx for char_idx,sent_idx in transcript_dict.items()}
    gt_starts_idx = [temp_transcript_dict[start[0]] for start in gt_starts]  # to sentence index
    gt_starts_idx.append(len(transcript_ls))  # index of last sentence + 1

    raw_response = response['raw_response']
    if raw_response == "Failed Request":
        return {"meeting_code": meeting_code, "error": "Failed Request"}
    
    llm_starts_temp = []
    try:
        response = response[response.index("["):] #peel the prefix text on number of segments
        llm_response = ast.literal_eval(response)
        if llm_response:
            for segment in llm_response:
                # remove all speaker labels for evaluation
                llm_starts_temp.append(re.sub(r'\s*SPEAKER_\d{2}:\s*', ' ', segment).strip())  # pure segmentation
            llm_nSegs = len(llm_response)
        else:
            raise Exception
    except Exception as e:
        raise ValueError(f"Invalid format in response for meeting {meeting_code}.")

    # adjust llm starting sentence to match full sentences
    llm_starts = []
    idx = -1
    for i in range(len(llm_starts_temp)):
        temp_start = llm_starts_temp[i]
        start = find_overlapping_llm_gt(temp_start.strip(), meeting_transcript_wo_speaker,
                                        list(indices[0] for indices in transcript_dict.keys()),
                                        idx, pt=False)  # return char index pairs

        if start:
            llm_starts.append(start)
            idx = start[1]
        else:
            print(f"Hallucination: meeting_code: {meeting_code}, \nsentence: {temp_start.strip()}")
            return {"meeting_code": meeting_code, "error": "Hallucination"}
    
    # starting sentence to sentence indexes in the transcript
    llm_starts_idx = [transcript_dict[start] for start in llm_starts]
    llm_starts_idx.append(len(transcript_ls)) # index of last sentence + 1

    output = {
        "meeting_code": meeting_code,
        "segment IDs": meeting_segIDs,
        "gt start indices": gt_starts_idx,
        "llm start indices": llm_starts_idx,
        "gt nSegs": gt_nSegs,
        "llm nSegs": llm_nSegs
    }
    return output

def transcript_2_sentences(transcript):
    """Break a transcript into a list of sentences."""
    # sentence_splitters = r'(?<=[.!?])\s+(?=[A-Z])|(?=SPEAKER_\d{2}:\s+)'
    sentence_splitters = r'(?<=[.!?])\s+(?=[A-Za-z0-9])'
    splits = re.split(sentence_splitters, transcript)

    merged_splits = []
    i = 0

    while i < len(splits):
        if splits[i].strip():  # Avoid adding empty strings
            merged_splits.append(splits[i].strip())
        i += 1

    return merged_splits

def extract_json(input_string):
    """Extract JSON from a raw LLM response string."""
    input_string = input_string.strip().replace('\n', '')
    json_match = re.search(r'\{.*', input_string)
    # if json_match:
    json_str = json_match.group(0)
    # print(json_str)
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        # print("Failed to parse JSON:", e)
        # print(json_str)
        return None

def find_overlapping_llm_gt(substring, long_string, transcript_indices, cur_idx=None, pt=False):
    """
    Find the sentence that the substring overlaps with within a transcript.
    returns start_index, end_index
    """
    Match = re.finditer(re.escape(substring), long_string)
    if pt == True: print(f"substring: {substring}, Match: {Match}")
    if Match:
        for match in Match:
            start_index = match.start()
            end_index = match.end()
            if pt == True: print(f"start: {start_index}, end: {end_index}")
            if cur_idx:
                if start_index < cur_idx:
                    if pt == True: print(f"Start index smaller than current index: {start_index}, {cur_idx}")
                    continue

            # Identify the first sentence that overlaps with the substring
            for i, cur_start in enumerate(transcript_indices):
                if pt == True: print(
                    f"i:{i}, cur_start:{cur_start}, len(matches): {len(transcript_indices)}, start_index: {start_index}")
                if i == len(transcript_indices) - 1:
                    break
                if cur_start <= start_index <= transcript_indices[i + 1]:
                    if pt == True: print(f"what I returned: {cur_start, transcript_indices[i + 1] - 1}")
                    return cur_start, transcript_indices[i + 1] - 1

    # Return None if no match is found
    if pt == True: print("Found no overlap in the transcript, substring:" + substring)
    return None

def find_overlapping_sentence(substring, long_string, cur_idx=None, pt=False):
    """
    Find the sentence that the substring overlaps with within a transcript.
    returns start_index, end_index
    """
    # print(f"cur index: {cur_idx}")
    Match = re.finditer(re.escape(substring), long_string)
    if pt==True: print(f"substring: {substring}, Match: {Match}")
    if Match:
        for match in Match:
            start_index = match.start()
            end_index = match.end()
            if pt==True: print(f"start: {start_index}, end: {end_index}")
            if cur_idx:
                if start_index < cur_idx:
                    if pt==True: print(f"Start index smaller than current index: {start_index}, {cur_idx}")
                    continue
            # match first chars in each sentence in the transcript and put them into a list
            matches = find_sentence_start(long_string)
            matches.append(len(long_string))
            matches = sorted(list(set(matches)))
            # print(f"sentence break matches: {matches}")

            # Identify the first sentence that overlaps with the substring
            for i, cur_start in enumerate(matches):
                if pt==True: print(f"i:{i}, cur_start:{cur_start}, len(matches): {len(matches)}, start_index: {start_index}")
                if i == len(matches)-1:
                    break
                if cur_start <= start_index < matches[i+1]:
                    if pt==True: print(f"what I returned: {cur_start, matches[i+1] - 1}")
                    return cur_start, matches[i+1] - 1

    # Return None if no match is found
    if pt==True: print("Found no overlap in the transcript, substring:" + substring)
    return None

def find_sentence_start(string):
    sentence_splitters = r'(^|(?<=[.!?;]))'
    # sentence_splitters_tag = r'(?=SPEAKER_\d{2}:)'
    temp_matches = list(re.finditer(sentence_splitters, string))
    # temp_matches.extend(re.finditer(sentence_splitters_tag, string))
    matches = []
    for m in temp_matches:
        m = m.start()
        if m >= len(string):
            continue
        flag = False
        while string[m].isspace():
            m += 1
            if m >= len(string):
                flag = True
                break
        if flag:
            continue
        matches.append(m)
    matches = sorted(list(set(matches)))

    return matches