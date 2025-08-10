from ntt import ntt, ntt_dit_rn, ntt_dif_nr, bit_rev_shuffle

class ArchitectureSimulator:
    def __init__(self, omegas, modulus, mem_latency, compute_latency, prefetch_latency=None, skip_compute=False):
        """
        Initialize the simulator with a provided NTT function.
        """
        self.pipeline = {
            "READ": (None, None),
            "COMPUTE": (None, None),
            "WRITE": (None, None)
        }
        self.omegas = omegas
        self.modulus = modulus
        self.out = None

        self.t_read = mem_latency
        self.t_compute = compute_latency
        self.t_write = mem_latency
        self.t_prefetch = prefetch_latency if prefetch_latency is not None else mem_latency
        self.debug = False
        self.skip_compute = skip_compute

        # initial cycle time of fetching the twiddle factors (omegas)
        # self.cycle_time = mem_latency
        self.cycle_time = 0

    def set_omegas(self, omegas):
        """
        Set the twiddle factors (omegas) for the NTT computation.
        """
        self.omegas = omegas
    
    def set_debug(self, debug):
        """
        Set the debug mode for the simulator.
        If True, print detailed information about each step.
        """
        self.debug = debug

    def prefetch(self):
        """
        Perform a prefetch operation, adding prefetch latency to cycle time.
        This should be called once before processing columns and once before processing rows.
        """
        self.cycle_time += self.t_prefetch

        if self.debug:
            print(f"Prefetch operation: added {self.t_prefetch} cycles. Total cycle time: {self.cycle_time}")
            print()

    def step(self, data, tag, tags_only=False):
        """
        Advance the simulator by one time step.
        Both `data` and `tag` are required and go into READ stage.

        Parameters:
        - data: For single PE: list of integers (single column)
                For multi-PE: list of lists where data[row][pe] contains the value for row and PE
        - tag: For single PE: single column index
               For multi-PE: list of column indices (one per PE)
        """
        if data is not None and not self.skip_compute:
            # Handle both single PE and multi-PE cases
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    # Multi-PE case: data[row][pe] format
                    num_pes = len(data[0])
                    num_rows = len(data)
                    # Verify consistent structure
                    for row in data:
                        if len(row) != num_pes:
                            raise ValueError("All rows must have the same number of PEs.")
                else:
                    # Single PE case: data is a list of integers (single column)
                    pass
            else:
                raise ValueError("Data must be a list of integers or list of lists, or None.")

        # Move COMPUTE result to WRITE
        self.pipeline["WRITE"] = self.pipeline["COMPUTE"]
        self.out = self.pipeline["WRITE"]

        # If there's something in read, apply NTT and put in COMPUTE
        if self.pipeline["READ"] != (None, None):
            read_data, read_tag = self.pipeline["READ"]
            
            if not self.skip_compute:

                if read_data is not None and isinstance(read_data, list) and len(read_data) > 0 and isinstance(read_data[0], list):
                    # Multi-PE case: convert data[row][pe] to separate columns and process each
                    num_pes = len(read_data[0])
                    num_rows = len(read_data)
                    
                    compute_results = []
                    for pe in range(num_pes):
                        # Extract column for this PE: data[row][pe] for all rows
                        column = [read_data[row][pe] for row in range(num_rows)]
                        # Apply NTT to this column
                        ntt_result = bit_rev_shuffle(ntt_dif_nr(column, self.modulus, self.omegas))
                        compute_results.append(ntt_result)
                    
                    compute_result = compute_results
                else:
                    # Single PE case: process single column
                    compute_result = bit_rev_shuffle(ntt_dif_nr(read_data, self.modulus, self.omegas))
            else:
                compute_result = read_data
            
            self.pipeline["COMPUTE"] = (compute_result, read_tag)
        else:
            self.pipeline["COMPUTE"] = (None, None)

        # Put new data and tag into read stage
        self.pipeline["READ"] = (data, tag)

        # Compute active stage times
        stage_times = []
        if self.pipeline["READ"] != (None, None):
            stage_times.append(self.t_read)
        else:
            stage_times.append(0)

        if self.pipeline["COMPUTE"] != (None, None):
            stage_times.append(self.t_compute)
        else:
            stage_times.append(0)

        if self.pipeline["WRITE"] != (None, None):
            stage_times.append(self.t_write)
        else:
            stage_times.append(0)

        # Add max active stage time to total cycle time
        cycles_elapsed = max(stage_times)
        self.cycle_time += max(stage_times)

        if self.debug:
            print(f"Cycle time: {self.cycle_time}, Cycles elapsed this step: {cycles_elapsed}")

            # Print the current state with tags only
            print(self.__str__(tags_only))
            print()

    def __str__(self, tags_only=False):
        """
        Nicely aligned output of all stages and their (tag, data).
        If tags_only is True, only print the tag for each stage.
        """
        col_widths = {
            "stage": 8,
            "tag": 10,
            "data": 20   # adjust if you want more space for data
        }

        def format_stage(stage):
            if self.pipeline[stage] == (None, None):
                tag_str = "None"
                data_str = "None"
            else:
                data, tag = self.pipeline[stage]
                
                # Handle both single column index and list of column indices
                if isinstance(tag, list):
                    # Multi-PE case: show list of column indices
                    tag_str = f"cols{tag}"
                else:
                    # Single PE case: show single column index
                    tag_str = f"col{tag}"
                
                # Handle both single PE and multi-PE cases for display
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    # Check if this is multi-PE input data (data[row][pe]) or multi-PE output data (data[pe][row])
                    if isinstance(tag, list):
                        # Multi-PE case: show number of PEs and rows
                        if len(data) > 0 and isinstance(data[0], list):
                            # Could be either data[row][pe] or data[pe][row]
                            data_str = f"[{len(data)}x{len(data[0])} matrix]"
                        else:
                            data_str = f"[{len(data)} cols]"
                    else:
                        data_str = f"[{len(data)} cols of size {len(data[0])}]"
                else:
                    # Single PE case: show as before but truncated
                    if data is not None and len(str(data)) > 15:
                        data_str = str(data)[:15] + "..."
                    elif data is not None:
                        data_str = str(data)
                    else:
                        data_str = "None"

            if tags_only:
                return (
                    f"{stage:<{col_widths['stage']}}: "
                    f"{tag_str:<{col_widths['tag']}}"
                )
            else:
                return (
                    f"{stage:<{col_widths['stage']}}: "
                    f"{tag_str:<{col_widths['tag']}} "
                    f"{data_str:<{col_widths['data']}}"
                )

        return " | ".join(format_stage(stage) for stage in ["READ", "COMPUTE", "WRITE"])