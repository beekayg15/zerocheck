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
    Return a dataframe sorted by the order of keys.
    """
    pattern = "|".join(map(re.escape, keys))
    result = df[df["Description"].str.contains(pattern, regex=True)]
    # Sort the result by the order of keys
    result_sort = deepcopy(result)
    result_sort["SortOrder"] = result["Description"].apply(
        lambda x: next(
            (i for i, key in enumerate(keys) if key in x), len(keys))
    )
    result = result_sort.sort_values("SortOrder").drop(columns=["SortOrder"])
    return result


def merge_dataframes(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, common_column: str, merge_column="Runtime (ms)"):
    """
    Assume both dataframes have one common column, merge their merge_column.
    Return df[↓common column, merge_column1, merge_column2]
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


def plot_univar_zc(file_path, start_line, save_fig=True):
    """
    Plot the univariate zerocheck average runtime from a log file.

    @param file_path: Path to the log file containing univariate zerocheck results.
    @param start_line: The starting line to identify the univariate zerocheck test blocks.
    @param save_fig: Whether to save the figure as a PNG file.
    @return: A dataframe containing the average runtime for each test case.
    """
    res_blocks = load_test_from_txt_to_blocks(file_path, start_line)

    target_keys_univar = [
        "Compute coset domain",
        "IFFT for g,h,s,o from evaluations to coefficients",
        "FFT Compute g,h,s,o,z,q evaluations over coset domain",
        "IFFT for q from evaluations to coefficients",
        "computing evaluations at challenge, get challenge",
        "Commit to (q) polynomial",
        # "Commit to (g,h,s,o) polynomials",
        # "Batch open the g,h,s,o,q poly commit at r",
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

    # Univariate ZC: merge all average results of all worksizes to one dataframe
    merge_average = worksize[min(worksize.keys())][["Description",
                                                    f"Average Runtime ({target_time_unit})"]].copy()
    for key, value in worksize.items():
        if key != min(worksize.keys()):
            merge_average = merge_dataframes(
                merge_average, value[["Description", f"Average Runtime ({target_time_unit})"]], "Description", f"Average Runtime ({target_time_unit})")

    for i, col in enumerate(merge_average.columns):
        if i == 0:
            continue
        merge_average.rename(
            columns={col: f"2^{list(worksize.keys())[i-1]} {col}"}, inplace=True)

    # draw stacked bar chart for each worksize, using the average runtime, stacked by rows
    runtime_columns = [
        col for col in merge_average.columns if f"Average Runtime ({target_time_unit})" in col]
    stacked_merge_average = merge_average.set_index("Description")[
        runtime_columns]
    stacked_merge_average = stacked_merge_average.T
    stacked_merge_average.plot(kind="bar", stacked=True, figsize=(10, 6))
    # rename the x-tick labels from 5 to the end
    plt.xticks(np.arange(len(runtime_columns)), [
               f"2^{i}" for i in range(min(worksize.keys()), min(worksize.keys())+len(runtime_columns))])
    plt.xlabel("Test Cases")
    # plt.yticks(np.arange(0, 30, 1))
    plt.ylabel(f"Average Runtime ({target_time_unit})")
    plt.title("Univariate Zerocheck Average Runtime")
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(title="Description", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_fig:
        print(f"Save figure to {file_path.replace('.log', '.png')}")
        plt.savefig(file_path.replace(".log", ".png"))
    return stacked_merge_average


def plot_multi_lin_zc(file_path, start_line, save_fig=True):
    """
    Plot the multilinear zerocheck average runtime from a log file.

    @param file_path: Path to the log file containing multilinear zerocheck results.
    @param start_line: The starting line to identify the multilinear zerocheck test blocks.
    @param save_fig: Whether to save the figure as a PNG file.
    @return: A dataframe containing the average runtime for each test case.
    """
    res_blocks = load_test_from_txt_to_blocks(file_path, start_line)

    target_keys_univar = [
        "computing inital challenge using which f_hat is computed",
        "Build MLE: computing f_hat(X) = sum_{B^m} f(X)",
        "running sumcheck proving algorithm for X rounds",
        "computing evaluations of input MLEs at challenges",
        # "commit to (g,h,s,o) input MLEs",
        # "batch open proof g,h,s,o input MLEs at r",
        ]
    assert len(target_keys_univar) == 4, "Result timer number not match the code"

    # parse the result from text log --> blocks --> a dataframe
    target_time_unit = "s"
    worksize = parse_text_to_dict_dataframe(
        res_blocks, target_keys_univar, target_time_unit)

    # for each worksize, we have a dataframe of results. Compute average value in columns containing "Runtime ()"
    for key, value in worksize.items():
        worksize[key] = dataframe_average_row(
            value, f"Runtime ({target_time_unit})")

    # MultiLin ZC: merge all average results of all worksizes to one dataframe
    merge_average = worksize[min(worksize.keys())][["Description",
                                                    f"Average Runtime ({target_time_unit})"]].copy()
    for key, value in worksize.items():
        if key != min(worksize.keys()):
            merge_average = merge_dataframes(
                merge_average, value[["Description", f"Average Runtime ({target_time_unit})"]], "Description", f"Average Runtime ({target_time_unit})")

    for i, col in enumerate(merge_average.columns):
        if i == 0:
            continue
        rename_goal = col if col.split()[-1] == "(s)" else " ".join(col.split()[:-1])
        rename_goal = f"2^{list(worksize.keys())[i-1]} {rename_goal}"
        merge_average.rename(columns={col: rename_goal}, inplace=True)

    # draw stacked bar chart for each worksize, using the average runtime, stacked by rows
    runtime_columns = [
        col for col in merge_average.columns if f"Average Runtime ({target_time_unit})" in col]
    stacked_merge_average = merge_average.set_index("Description")[
        runtime_columns]
    stacked_merge_average = stacked_merge_average.T
    stacked_merge_average.plot(kind="bar", stacked=True, figsize=(10, 6))
    # rename the x-tick labels from 5 to the end
    plt.xticks(np.arange(len(runtime_columns)), [
               f"2^{i}" for i in range(min(worksize.keys()), min(worksize.keys())+len(runtime_columns))])
    plt.xlabel("Test Cases")
    # plt.yticks(np.arange(0, 30, 1))
    plt.ylabel(f"Average Runtime ({target_time_unit})")
    plt.title("Multilinear Zerocheck Average Runtime")
    plt.grid(True, axis='y', linestyle='--')
    plt.legend(title="Description", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_fig:
        print(f"Save figure to {file_path.replace('.log', '.png')}")
        plt.savefig(file_path.replace(".log", ".png"))
    return stacked_merge_average


def parallel_stacked_bar_chart(dfs: list, output_path=None):
    """
    Draw a parallel stacked bar chart for multiple dataframes.
    Each dataframe represents a stacked bar chart, and the bars are grouped by the union of all row indices.

    @param dfs: List of dataframes to plot. Each dataframe should have row indices representing x-axis labels.
    @param output_path: Path to save the figure if save_fig is True.
    """
    color_set = ["#C70E7BFF", "#FC6882FF", "#007BC3FF", "#54BCD1FF", "#EF7C12FF", "#F4B95AFF", "#009F3FFF",
                 "#8FDA04FF", "#AF6125FF", "#F4E3C7FF", "#B25D91FF", "#EFC7E6FF", "#EF7C12FF", "#F4B95AFF"]
    # Find the union of all row indices
    all_indices = sorted(set().union(*[df.index for df in dfs]))
    num_dfs = len(dfs)
    bar_width = 0.8 / num_dfs  # Divide the space for bars equally
    x = np.arange(len(all_indices))  # Base x positions for the groups

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, df in enumerate(dfs):
        # Align the dataframe with the full index set, filling missing values with 0
        df_aligned = df.reindex(all_indices, fill_value=0)
        # Offset x positions for each dataframe
        x_offset = x + i * bar_width
        # Plot stacked bars for the current dataframe
        bottom = np.zeros(len(all_indices))
        for j, col in enumerate(df_aligned.columns):
            if (i == 0 and j < 6) or (i == 1 and j < 4):
                ax.bar(x_offset, df_aligned[col], bar_width, label=f"{col}", bottom=bottom, color=color_set[j % len(
                    color_set)], )  # hatch='.'
            else:
                ax.bar(x_offset, df_aligned[col], bar_width,
                       label=f"{col}", bottom=bottom, color=color_set[j % len(color_set)])
            bottom += df_aligned[col].values

    # Set x-axis labels and ticks
    ax.set_xticks(x + (num_dfs - 1) * bar_width / 2)
    ax.set_xticklabels([i.split()[0] for i in all_indices], rotation=45)
    ax.set_ylabel("Runtime (s)")
    # Set y-axis ticks to show each 5% of total height
    max_height = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, max_height + max_height * 0.05, max_height * 0.05))
    ax.grid(axis='y', linestyle='--')

    # Add dummy labels for legend
    nrow_first_column_legend = len(dfs[0].columns)
    dummy_labels = [" " for _ in range(
        nrow_first_column_legend * 2 - (len(dfs[0].columns) + len(dfs[1].columns)))]
    for label in dummy_labels:
        ax.bar(0, 0, color='none', label=label)
    # Set legend
    ax.legend(bbox_to_anchor=(1, 1.3), ncol=2)

    plt.tight_layout()
    if output_path:
        print(f"Save figure to {output_path}")
        plt.savefig(output_path)


def parallel_stacked_bar_chart_univ_mulin_numvar_comm(save_fig=False):
    """
    Draw a parallel stacked bar chart for univariate and multilinear ZC, 
    different num_var, different commit methods.
    """
    # extract the averaged data from log files, return the dataframe
    stack_multilin_zc_hyrax_df = plot_multi_lin_zc(
        # file_path="output_log/mullin_opt_bench_24_24_run_1_hyrax_open_4.log",
        file_path="output_log/mullin_opt_bench_24_24_run_12_hyrax_open_48.log",
        start_line="Prover starts Opt multilinear for 2^",
        save_fig=False,
    )
    stack_multilin_zc_kzg_df = plot_multi_lin_zc(
        # file_path="output_log/mullin_opt_bench_24_24_run_1_kzg_open_4.log",
        file_path="output_log/mullin_opt_bench_24_24_run_12_kzg_open_48.log",
        start_line="Prover starts Opt multilinear for 2^",
        save_fig=False,
    )
    stack_multilin_zc_ligeroPoseidon_df = plot_multi_lin_zc(
        file_path="output_log/mullin_opt_bench_24_24_run_12_ligeroposeidon_open_48.log",
        start_line="Prover starts Opt multilinear for 2^",
        save_fig=False,
    )
    stack_multilin_zc_ligero_df = plot_multi_lin_zc(
        # file_path="output_log/mullin_opt_bench_24_24_run_1_ligero_open_4.log",
        file_path="output_log/mullin_opt_bench_24_24_run_12_ligero_open_48.log",
        start_line="Prover starts Opt multilinear for 2^",
        save_fig=False,
    )
    stack_multilin_zc_hyrax_df.index = stack_multilin_zc_hyrax_df.index.str.replace(
        "Average Runtime", "Mulin hyrax")
    stack_multilin_zc_kzg_df.index = stack_multilin_zc_kzg_df.index.str.replace(
        "Average Runtime", "Mulin kzg")
    stack_multilin_zc_ligero_df.index = stack_multilin_zc_ligero_df.index.str.replace(
        "Average Runtime", "Mulin ligero")
    stack_multilin_zc_ligeroPoseidon_df.index = stack_multilin_zc_ligeroPoseidon_df.index.str.replace(
        "Average Runtime", "Mulin ligeroPoseidon")
    
    # extract the averaged data from log files, return the dataframe
    stack_univar_zc_kzg_df = plot_univar_zc(
        # file_path="output_log/univar_opt_bench_24_24_run_1_kzg_open_5.log",
        file_path="output_log/univar_opt_bench_24_24_run_12_kzg_open_60.log",
        start_line="Opt Univariate Proof Generation Test ",
        save_fig=False,
    )
    stack_univar_zc_ligeroPoseidon_df = plot_univar_zc(
        file_path="output_log/univar_opt_bench_24_24_run_12_ligeroposeidon_open_60.log",
        start_line="Opt Univariate Proof Generation Test ",
        save_fig=False,
    )
    stack_univar_zc_ligero_df = plot_univar_zc(
        # file_path="output_log/univar_opt_bench_24_24_run_1_ligero_open_5.log",
        file_path="output_log/univar_opt_bench_24_24_run_12_ligero_open_60.log",
        start_line="Opt Univariate Proof Generation Test ",
        save_fig=False,
    )
    stack_univar_zc_kzg_df.index = stack_univar_zc_kzg_df.index.str.replace(
        "Average Runtime", "Univar kzg")
    stack_univar_zc_ligero_df.index = stack_univar_zc_ligero_df.index.str.replace(
        "Average Runtime", "Univar ligero")
    stack_univar_zc_ligeroPoseidon_df.index = stack_univar_zc_ligeroPoseidon_df.index.str.replace(
        "Average Runtime", "Univar ligeroPoseidon")

    # merge the dataframe, match the same columns between dataframes to the correct column
    merged_multilin_df = pd.concat(
        [stack_multilin_zc_hyrax_df, 
         stack_multilin_zc_kzg_df, 
        #  stack_multilin_zc_ligeroPoseidon_df, 
         stack_multilin_zc_ligero_df
         ], axis=0)
    merged_univar_df = pd.concat(
        [stack_univar_zc_kzg_df, 
        #  stack_univar_zc_ligeroPoseidon_df, 
         stack_univar_zc_ligero_df
         ], axis=0)
    
    # sort the order by the index column
    merged_multilin_df = merged_multilin_df.sort_index()
    merged_univar_df = merged_univar_df.sort_index()

    # Draw a stacked bar chart for merged_multilin_df and merged_univar_df
    combined_df = pd.concat([merged_multilin_df, merged_univar_df], axis=1).fillna(0)

    # Plot the stacked bar chart
    combined_df.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="tab20")

    plt.xlabel("Test Cases")
    plt.ylabel("Runtime (s)")
    plt.title("Stacked Bar Chart for Multilinear and Univariate ZC")
    plt.xticks(rotation=45)

    # Split the legend into two groups
    group_size = 4
    handles, labels = plt.gca().get_legend_handles_labels()
    first_legend_handles = handles[:group_size]
    first_legend_labels = labels[:group_size]
    second_legend_handles = handles[group_size:]
    second_legend_labels = labels[group_size:]

    # Add the first legend
    first_legend = plt.legend(first_legend_handles, first_legend_labels, loc='upper left', bbox_to_anchor=(1, 1), title="Mulin")
    plt.gca().add_artist(first_legend)  # Add the first legend to the plot

    # Add the second legend
    plt.legend(second_legend_handles, second_legend_labels, loc='upper left', bbox_to_anchor=(1, 0.7), title="Univar")
    plt.tight_layout()

    if save_fig:
        output_path = "output_log/stacked_bar_chart_univ_mulin_bench_24_24_run12_filter.png"
        print(f"Save figure to {output_path}")
        plt.savefig(output_path)

    return None


if __name__ == '__main__':

    save_each_fig = False

    # stack_univar_zc_df = plot_univar_zc(
    #     file_path="output_log/univar_opt_bench_24_24_run_1_kzg_open_5.log",
    #     start_line="Opt Univariate Proof Generation Test ",
    #     save_fig=save_each_fig,
    # )

    # stack_multi_lin_zc_df = plot_multi_lin_zc(
    #     file_path="output_log/mullin_opt_bench_24_24_run_1_kzg_open_4.log",
    #     start_line="Prover starts Opt multilinear for 2^",
    #     save_fig=save_each_fig,
    # )

    # # Parallel stacked bar chart for both univariate and multilinear ZC
    # parallel_stacked_bar_chart(
    #     [stack_univar_zc_df, stack_multi_lin_zc_df],
    #     output_path="output_log/parallel_stacked_bar_chart_1_22_26.png"
    # )

    parallel_stacked_bar_chart_univ_mulin_numvar_comm(True)

    print("End...")
