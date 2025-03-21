import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import re


def load_test_from_txt_to_blocks(file_path: str, start_line: str):
    """
    read the txt, return a list of blocks that a block(list) 
    contain one test (from Start: start_line to ··End:).
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

    result_blocks = []
    block = []
    title = ""
    for i, line in enumerate(lines):
        if line.startswith("Start:") and start_line in line:
            title = line.replace("Start:", "").strip()
            block = [line]
            continue
        if line.startswith("··End:") and start_line in line:
            assert title in line, f"Title not match: {title} vs {line}"
            block.append(line)
            result_blocks.append(deepcopy(block))
            continue
        else:
            block.append(line)

    return result_blocks


def convert_time(time_str: str, target_unit: str):
    """Convert time values to target unit (s, ms, us, ns)."""
    match = re.match(r"([\d.]+)\s*(ms|µs|ns|s)", time_str)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    conversion_factors = {"s": 1000, "ms": 1,
                          "µs": 0.001, "ns": 0.000001, "us": 0.001}
    result = value * conversion_factors[unit] / conversion_factors[target_unit]
    return result


def parse_test_block_time(block: list, target_time_unit: str = "ms"):
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
            runtime = convert_time(runtime_str + line[-2:], target_time_unit)

            # re.split(r'End:|\.\.+| \.', line[:i+1])
            parts = line[:i+1].replace(".", "").split("End:")
            parts = [re.sub(r"\s+", " ", part).strip()
                     for part in parts if part.strip()]
            description = parts[1].strip()

            if runtime is not None:
                results.append(
                    {"Description": description, f"Runtime ({target_time_unit})": runtime})
            else:
                raise ValueError(f"Invalid runtime: {line}")
    result_df = pd.DataFrame(results)
    return result_df


def extract_result_by_key(df: pd.DataFrame, keys: list):
    """
    Extract the result of a test with a list of keys (should be unique each).
    Return a dataframe.
    """
    pattern = "|".join(map(re.escape, keys))
    result = df[df["Description"].str.contains(pattern, regex=True)]
    return result


def merge_dataframes(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, common_column: str, merge_column="Runtime (ms)"):
    """
    Assume both dataframes have one common column, merge their merge_column.
    """
    assert np.array_equal(dataframe1[common_column].values, dataframe2[common_column]
                          .values), f"Common column not match, {dataframe1[common_column]} vs {dataframe2[common_column]}"
    runtime_columns = [
        col for col in dataframe1.columns if merge_column in col]
    new_suffix = len(runtime_columns)
    dataframe2 = dataframe2.rename(
        columns={merge_column: f"{merge_column} {new_suffix}"})
    merged = pd.merge(dataframe1, dataframe2, on=common_column)
    return merged


def parse_text_to_dict_dataframe(text_list: list, target_keys: list = [], target_time_unit: str = "ms") -> dict:
    """
    Parse the text lines (as list) to a dictionary of dataframes.
    Repeat tests are merged into one dataframe.

    Parse the result from text log --> blocks --> a dataframe.

    @param text_list: list of all text lines from .log including repeated tests. 
    @param target_keys: filter the focus result by target keys.
    @param target_time_unit: convert time values to target unit (s, ms, us, ns).
    @return: {worksize 2^x: (dataframe) result of repeated tests}
    """
    worksize = {}  # {worksize 2^x: result of repeated tests}
    for idx, block in enumerate(text_list):
        # A block result of one test
        block_result = parse_test_block_time(block, target_time_unit)

        # Workload Size: extract the number between "for 2^" and "work"
        focus_result_title = block_result.iloc[-1]["Description"]
        focus_result_size = int(
            focus_result_title.split("for 2^")[1].split("work")[0])

        # Filter the focus result by target keys
        if target_keys:
            focus_result = extract_result_by_key(
                block_result, target_keys)
            assert len(focus_result) == len(
                target_keys), f"Result timer number not match the code, only filter out {len(focus_result)}"
        else:
            focus_result = block_result

        # Merge the repeating test to one dataframe, then save to worksize
        if focus_result_size not in worksize:
            worksize[focus_result_size] = focus_result
        else:
            worksize[focus_result_size] = merge_dataframes(
                worksize[focus_result_size], focus_result, "Description", f"Runtime ({target_time_unit})")
    return worksize


def dataframe_average_row(dataframe: pd.DataFrame, common_column_name: str = "Runtime (ms)"):
    """
    Compute the average value of each row, columns contain "common_column_name" 
    in a dataframe.
    """
    runtime_columns = [
        col for col in dataframe.columns if common_column_name in col]
    dataframe[f"Average {common_column_name}"] = dataframe[runtime_columns].mean(
        axis=1)
    return dataframe


def plot_univar_zc():
    res_blocks = load_test_from_txt_to_blocks(
        "output_log/univar_opt_bench_multhr_1.log", "Opt Univariate Proof Generation Test for")

    target_keys_univar = ["IFFT for g,h,s,o from evaluations to coefficients",
                        #   "Setup KZG10 polynomial commitments global parameters",
                        #   "Setup verifier key",
                          "KZG commit to (g,h,s,o) polynomials",
                          "Get Fiat-Shamir random challenge and evals at challenge",
                          "KZG open the g,h,s,o poly commit at r",
                          "Compute coset domain",
                          "Compute g,h,s,o,z,q evaluations over coset domain",
                          "IFFT for q from evaluations to coefficients",
                          "KZG commit to q polynomial",
                          "KZG open the q poly commit at r",]
    assert len(target_keys_univar) == 9, "Result timer number not match the code"

    # parse the result from text log --> blocks --> a dataframe
    target_time_unit = "s"
    worksize = parse_text_to_dict_dataframe(
        res_blocks, target_keys_univar, target_time_unit)

    # for each worksize, we have a dataframe of results. Compute average value in columns containing "Runtime ()"
    for key, value in worksize.items():
        worksize[key] = dataframe_average_row(
            value, f"Runtime ({target_time_unit})")

    # Univariate ZC: merge all average results of all worksizes to one dataframe
    merge_average = worksize[5][["Description",
                                 f"Average Runtime ({target_time_unit})"]].copy()
    for key, value in worksize.items():
        if key != 5:
            merge_average = merge_dataframes(
                merge_average, value[["Description", f"Average Runtime ({target_time_unit})"]], "Description", f"Average Runtime ({target_time_unit})")

    # draw stacked bar chart for each worksize, using the average runtime, stacked by rows
    runtime_columns = [
        col for col in merge_average.columns if f"Average Runtime ({target_time_unit})" in col]
    stacked_merge_average = merge_average.set_index("Description")[
        runtime_columns]
    stacked_merge_average = stacked_merge_average.T
    stacked_merge_average.plot(kind="bar", stacked=True, figsize=(10, 6))
    # rename the x-tick labels from 5 to the end
    plt.xticks(np.arange(len(runtime_columns)), [
               f"2^{i}" for i in range(5, 5+len(runtime_columns))])
    plt.xlabel("Test Cases")
    # plt.yticks(np.arange(0, 30, 1))
    plt.ylabel(f"Average Runtime ({target_time_unit})")
    plt.title("Univariate Zerocheck Average Runtime")
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(title="Description", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output_log/univar_opt_bench_multhr_1.png")


def plot_multi_lin_zc():
    res_blocks = load_test_from_txt_to_blocks(
        "output_log/mullin_opt_bench_multhr_1.log", "Prover starts Opt multilinear for 2^")

    target_keys_univar = ["commit to (g,h,s,o) input MLEs",
                          "computing inital challenge using which f_hat is computed",
                          "Build MLE: computing f_hat(X) = sum_{B^m} f(X)",
                          "running sumcheck proving algorithm for X rounds",
                          "open proof g,h,s,o input MLEs at r",
                          "computing evaluations of input MLEs at challenges",
                          ]
    assert len(target_keys_univar) == 6, "Result timer number not match the code"

    # parse the result from text log --> blocks --> a dataframe
    target_time_unit = "s"
    worksize = parse_text_to_dict_dataframe(
        res_blocks, target_keys_univar, target_time_unit)

    # for each worksize, we have a dataframe of results. Compute average value in columns containing "Runtime ()"
    for key, value in worksize.items():
        worksize[key] = dataframe_average_row(
            value, f"Runtime ({target_time_unit})")

    # MultiLin ZC: merge all average results of all worksizes to one dataframe
    merge_average = worksize[5][["Description",
                                 f"Average Runtime ({target_time_unit})"]].copy()
    for key, value in worksize.items():
        if key != 5:
            merge_average = merge_dataframes(
                merge_average, value[["Description", f"Average Runtime ({target_time_unit})"]], "Description", f"Average Runtime ({target_time_unit})")

    # draw stacked bar chart for each worksize, using the average runtime, stacked by rows
    runtime_columns = [
        col for col in merge_average.columns if f"Average Runtime ({target_time_unit})" in col]
    stacked_merge_average = merge_average.set_index("Description")[
        runtime_columns]
    stacked_merge_average = stacked_merge_average.T
    stacked_merge_average.plot(kind="bar", stacked=True, figsize=(10, 6))
    # rename the x-tick labels from 5 to the end
    plt.xticks(np.arange(len(runtime_columns)), [
               f"2^{i}" for i in range(5, 5+len(runtime_columns))])
    plt.xlabel("Test Cases")
    # plt.yticks(np.arange(0, 30, 1))
    plt.ylabel(f"Average Runtime ({target_time_unit})")
    plt.title("Multilinear Zerocheck Average Runtime")
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(title="Description", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output_log/mullin_opt_bench_multhr_1.png")


if __name__ == '__main__':
    # plot_univar_zc()
    plot_multi_lin_zc()

    print("End...")
