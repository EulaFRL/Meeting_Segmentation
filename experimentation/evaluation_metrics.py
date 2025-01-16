import json
import math
import statistics
from typing import List

def evaluation(jsonl_file_path):
    scores = []
    error_count = [0, 0, 0]  # failed requests, json parse errors, hallucination errors

    with open(jsonl_file_path, 'r') as file:
        json_lines = [json.loads(line) for line in file]

    for piece in json_lines:
        if "error" in piece.keys():
            if piece["error"] == "Failed Request":
                error_count[0] += 1
            elif piece["error"] == "Format Error":
                error_count[1] += 1
            elif piece["error"] == "Hallucination":
                error_count[2] += 1
            
            continue
        
        llm_indices = piece['llm start indices']
        gt_indices = piece['gt start indices']

        # Skip invalid cases
        if piece['llm nSegs'] == 1:
            piece['seg score'] = 0
            scores.append(0)
            continue

        # Calculate overlap score
        scores_by_segment = {}
        for i in range(len(llm_indices) - 1):
            max_score = -1
            for j in range(len(gt_indices) - 1):
                cur_score = segment_overlap_score_A((gt_indices[j], gt_indices[j + 1]),
                                                    (llm_indices[i], llm_indices[i + 1]))
                if cur_score > max_score:
                    max_score = cur_score
            scores_by_segment[i] = max_score
        overlap_score = sum(scores_by_segment.values()) / piece['llm nSegs']

        # Over/under-segmentation penalty
        nSeg_penalty = abs(piece['gt nSegs'] - piece['llm nSegs']) / piece['gt nSegs']

        # Starting sentence penalty
        start_penalty = llm_indices[0] / (llm_indices[1] if llm_indices[1] != 0 else 1)

        # Final piece score
        piece_score = overlap_score * (1 - sigmoid_scale(nSeg_penalty)) * (1 - sigmoid_scale(start_penalty))

        # Append the score and update the piece
        scores.append(piece_score)
        piece['seg score'] = piece_score

    # Write back updated JSONL file
    with open(jsonl_file_path, 'w') as file:
        for piece in json_lines:
            file.write(json.dumps(piece) + '\n')

    # Generate report
    generate_report(scores, error_count, len(json_lines))

def sigmoid_scale(x):
    return (2 / (1 + math.exp(-x))) - 1

def segment_overlap_score_A(gt_segment, llm_segment):
    """
    Calculates the F1 score for the overlap between ground truth (GT) and LLM segments.
    """
    if gt_segment[0] <= llm_segment[0] < gt_segment[1]:
        true_positive = min(gt_segment[1] - llm_segment[0], llm_segment[1] - llm_segment[0])
    elif llm_segment[0] <= gt_segment[0] < llm_segment[1]:
        true_positive = min(llm_segment[1] - gt_segment[0], gt_segment[1] - gt_segment[0])
    else:
        return 0

    precision = true_positive / (llm_segment[1] - llm_segment[0])
    recall = true_positive / (gt_segment[1] - gt_segment[0])
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def generate_report(scores: List[float], error_count: List[int], total_pieces: int):
    """
    Generate a segmentation performance report.
    """
    avg = sum(scores) / (total_pieces + error_count[1] + error_count[2])
    avg_no_error = sum(scores) / len(scores)
    min_no_error = min(scores)
    max_no_error = max(scores)
    std_no_error = statistics.stdev(scores) if len(scores) > 1 else 0

    for _ in range(error_count[1] + error_count[2]):
        scores.append(0)
    std_with_error = statistics.stdev(scores) if len(scores) > 1 else 0

    report = (
        f"Total pieces: {total_pieces}\n"
        f"Failed requests: {error_count[0]}\n"
        f"Format errors: {error_count[1]}\n"
        f"Memory errors: {error_count[2]}\n"
        f"Average score (including errors): {avg:.4f}\n"
        f"Average score (excluding errors): {avg_no_error:.4f}\n"
        f"Min score (excluding errors): {min_no_error:.4f}\n"
        f"Max score: {max_no_error:.4f}\n"
        f"Standard deviation (excluding errors): {std_no_error:.4f}\n"
        f"Standard deviation (including errors): {std_with_error:.4f}\n"
    )

    print(report)
    return report
