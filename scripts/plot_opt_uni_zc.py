import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import re


def load_test_from_txt_to_blocks(file_path):
    """
    read the txt, return a list of blocks that a block(list) 
    contain one test (from Start: to ··End:).
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

    result_blocks = []
    block = []
    title = ""
    for i, line in enumerate(lines):
        if line.startswith("Start:"):
            title = line.replace("Start:", "").strip()
            block = [line]
            continue
        if line.startswith("··End:"):
            assert title in line, f"Title not match: {title} vs {line}"
            block.append(line)
            result_blocks.append(deepcopy(block))
            continue
        else:
            block.append(line)

    return result_blocks


def convert_to_ms(time_str):
    """Convert time values to milliseconds."""
    match = re.match(r"([\d.]+)\s*(ms|µs|ns|s)", time_str)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)

    conversion_factors = {"s": 1000, "ms": 1, "µs": 0.001, "ns": 0.000001}
    return value * conversion_factors[unit]


def parse_test_block_ms(block):
    """
    parse a block of test result runtime to a dataframe.
    """
    results = []

    for line in block:
        if "·End:" in line:
            runtime_str = ""
            dot_seen = False
            i = len(line) - 1 - 2
            while i >= 0:
                char = line[i]
                if char.isdigit():
                    runtime_str = char + runtime_str  # Prepend digit
                elif char == ".":
                    if dot_seen or i == 0 or not line[i - 1].isdigit():
                        break  # Stop if we already saw a dot OR if it's not part of a float
                    dot_seen = True
                    # Prepend dot (part of a float)
                    runtime_str = char + runtime_str
                else:
                    break  # Stop on any non-numeric, non-dot character
                i -= 1
            runtime = convert_to_ms(runtime_str + line[-2:])

            # re.split(r'End:|\.\.+| \.', line[:i+1])
            parts = line[:i+1].replace(".", "").split("End:")
            parts = [re.sub(r"\s+", " ", part).strip()
                     for part in parts if part.strip()]
            description = parts[1].strip()

            if runtime is not None:
                results.append(
                    {"Description": description, "Runtime (ms)": runtime})
            else:
                raise ValueError(f"Invalid runtime: {line}")
    result_df = pd.DataFrame(results)
    return result_df


def extract_result_by_key(df, keys):
    """
    Extract the result of a test with a list of keys (should be unique each).
    Return a dataframe.
    """
    pattern = "|".join(map(re.escape, keys))
    result = df[df["Description"].str.contains(pattern, regex=True)]
    return result


def merge_dataframes(dataframe1, dataframe2, common_column, merge_column="Runtime (ms)"):
    """
    Assume both dataframes have one common column, merge their merge_column.
    """
    assert np.array_equal(dataframe1[common_column].values, dataframe2[common_column]
                          .values), f"Common column not match, {dataframe1[common_column]} vs {dataframe2[common_column]}"
    runtime_columns = [col for col in dataframe1.columns if merge_column in col]
    new_suffix = len(runtime_columns)
    dataframe2 = dataframe2.rename(columns={merge_column: f"{merge_column} {new_suffix}"})
    merged = pd.merge(dataframe1, dataframe2, on=common_column)
    return merged


if __name__ == '__main__':
    # res_blocks = load_test_from_txt("scripts/new.txt")
    res_blocks = load_test_from_txt_to_blocks(
        "output_log/output_opt_uni_zc_tests_multhr_1.log")

    target_keys_univar = ["IFFT for g,h,s,o from evaluations to coefficients",
                          "Setup KZG10 polynomial commitments global parameters",
                          "Setup verifier key",
                          "KZG commit to (g,h,s,o) polynomials",
                          "Get Fiat-Shamir random challenge and evals at challenge",
                          "KZG open the g,h,s,o poly commit at r",
                          "Compute coset domain",
                          "Compute g,h,s,o,z,q evaluations over coset domain",
                          "IFFT for q from evaluations to coefficients",
                          "KZG commit to q polynomial",
                          "KZG open the q poly commit at r",]

    worksize = {}  # key: worksize 2^x, value: result of repeated tests
    for idx, block in enumerate(res_blocks):
        # A block result of one test
        block_result = parse_test_block_ms(block)
        total_prove_time = extract_result_by_key(
            block_result, ["OptimizedUnivariateZeroCheck::prove, with"])
        # print(total_prove_time)
        focus_result_title = block_result.iloc[-1]["Description"]
        # extract the number between "for 2^" and "work"
        focus_result_size = int(
            focus_result_title.split("for 2^")[1].split("work")[0])
        focus_result = extract_result_by_key(block_result, target_keys_univar)
        assert len(focus_result) == 11, "Result timer number not match the code"
        # print(f"sum of focus result (ms): {focus_result['Runtime (ms)'].sum()}")
        if focus_result_size not in worksize:
            worksize[focus_result_size] = focus_result
        else:
            worksize[focus_result_size] = merge_dataframes(
                worksize[focus_result_size], focus_result, "Description")

    # for each worksize, we have a dataframe of results. Compute a column of average runtime
    for key, value in worksize.items():
        # compute the average value in column containing "Runtime (ms)"
        runtime_columns = [col for col in value.columns if "Runtime (ms)" in col]
        value[f"Average Runtime (ms)"] = value[runtime_columns].mean(axis=1)

    merge_average = worksize[5][["Description", "Average Runtime (ms)"]].copy()
    for key, value in worksize.items():
        if key != 5:
            merge_average = merge_dataframes(
                merge_average, value[["Description", "Average Runtime (ms)"]], "Description", "Average Runtime (ms)")

    # draw stacked bar chart for each worksize, using the average runtime, stacked by rows
    runtime_columns = [col for col in merge_average.columns if "Average Runtime (ms)" in col]
    stacked_merge_average = merge_average.set_index("Description")[runtime_columns]
    stacked_merge_average = stacked_merge_average.T
    stacked_merge_average.plot(kind="bar", stacked=True, figsize=(10, 6))
    # rename the x-tick labels from 5 to the end
    plt.xticks(np.arange(len(runtime_columns)), [f"2^{i}" for i in range(5, 5+len(runtime_columns))])
    plt.xlabel("Test Cases")
    plt.ylabel("Average Runtime (ms)")
    plt.title("Univariate Zerocheck Average Runtime")
    plt.legend(title="Description", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("End...")
