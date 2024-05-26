import json
import os
import pandas as pd
import re

from collections import defaultdict

PWD = os.environ["WORKSPACE_PATH"]

MBPT = "MBPT"
BIG_5 = "BIG 5"
LABEL = "LABEL"
VOTES = "VOTES"

MBPT_REGEX = r"[EI][NS][TF][JP]"
BIG_5_REGEX = r"[SR][CL][OU][EA][IN]"

pers_defs = {
    MBPT: {
        0: (("I", "introverted"), ("E", "extroverted")),
        1: (("S", "sensing"), ("N", "intuitive")),
        2: (("F", "feeling"), ("T", "thinking")),
        3: (("J", "judging"), ("P", "perceiving")),
    },
    BIG_5: {
        0: (("S", "social"), ("R", "reserved")),
        1: (("L", "limbic"), ("C", "calm")),
        2: (("O", "organized"), ("U", "unstructured")),
        3: (("A", "agreeable"), ("E", "egocentric")),
        4: (("N", "non-curious"), ("I", "inquisitive")),
    }
}

def get_label_mode(data_type) -> str:
    # data_type = <personality_type>_<dimension_number>_<label_type>
    return MBPT if MBPT.lower() in data_type else BIG_5

def get_labels_and_defs(data_type):
    label_mode = get_label_mode(data_type)
    index = [idx for idx in range(5) if str(idx) in data_type][0]
    return pers_defs[label_mode][index]

def get_pers_df():
    with open(f"{PWD}/data/personality_data/char_to_pers_votes.json", "r+") as fp:
        pers_rows = json.load(fp)
        new_pers_rows = []
        for char_id in pers_rows:
            pers_votes = pers_rows[char_id]
            new_pers_votes = {"char_id": char_id}
            new_pers_votes["MBPT"] = pers_votes["Myers Briggs"]
            new_pers_votes["BIG 5"] = pers_votes["SLOAN"]
            new_pers_rows.append(new_pers_votes)
        pers_df = pd.DataFrame(new_pers_rows)
    return pers_df


def get_dim_labels(row, pers_mode, min_votes=1):
    """
    produce labels per personality dimension
    """
    pers_reg = BIG_5_REGEX if pers_mode == BIG_5 else MBPT_REGEX
    pers_votes = sorted(row[pers_mode].items(), key=lambda x: x[1], reverse=True)
    top_pers = pers_votes[0][0]
    if re.match(pers_reg, top_pers):
        if min_votes > sum([vote[1] for vote in pers_votes]):
            return None
        else:
            return list(top_pers)
    elif "X" in top_pers:
        return None
    raise Exception(f"Invalid personality type ({top_pers})!")


def get_alt_dim_labels(row, pers_mode):
    """
    produce alt labels per personality dimension
    """
    pers_reg = BIG_5_REGEX if pers_mode == BIG_5 else MBPT_REGEX
    pers_len = 5 if pers_mode == BIG_5 else 4
    dim_votes = [defaultdict(int) for _ in range(pers_len)]
    for pers, count in row[pers_mode].items():
        if re.match(pers_reg, pers):
            for idx, dim in enumerate(pers):
                dim_votes[idx][dim] += count
        elif "X" in pers:
            continue
        else:
            raise Exception(f"Invalid personality type ({pers})!")
    labels = []
    for idx, dv in enumerate(dim_votes):
        if len(dv) > 0:
            top_dim = sorted(dv.items(), key=lambda x: x[1], reverse=True)[0][0]
            labels.append(top_dim)
    return labels if labels else None


def get_labels(row, pers_mode):
    """
    produce labels per personality
    """
    pers_reg = BIG_5_REGEX if pers_mode == BIG_5 else MBPT_REGEX
    pers_votes = sorted(row[pers_mode].items(), key=lambda x: x[1], reverse=True)
    top_pers = pers_votes[0][0]
    if re.match(pers_reg, top_pers):
        return top_pers
    elif "X" in top_pers:
        return None
    raise Exception(f"Invalid personality type ({top_pers})!")


def get_dim_votes(row, pers_mode):
    """
    produce labels by personality dimension votes
    """
    pers_reg = BIG_5_REGEX if pers_mode == BIG_5 else MBPT_REGEX
    pers_len = 5 if pers_mode == BIG_5 else 4
    votes = [list() for _ in range(pers_len)]
    for pers, count in row[pers_mode].items():
        if re.match(pers_reg, pers):
            for idx, dim in enumerate(pers):
                votes[idx].extend([dim for _ in range(count)])
        elif "X" in pers:
            continue
        else:
            raise Exception(f"Invalid personality type ({pers})!")
    return votes


def get_votes(row, pers_mode):
    """
    produce labels by personality votes
    """
    pers_reg = BIG_5_REGEX if pers_mode == BIG_5 else MBPT_REGEX
    votes = []
    for pers, count in row[pers_mode].items():
        if re.match(pers_reg, pers):
            votes.extend([pers for _ in range(count)])
        elif "X" in pers:
            continue
        else:
            raise Exception(f"Invalid personality type ({pers})!")
    return votes
