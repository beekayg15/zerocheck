from ntt import ntt, ntt_dit_rn, ntt_dif_nr, bit_rev_shuffle

class ArchitectureSimulator:
    def __init__(self, omegas, modulus, mem_latency, compute_latency):
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

        # initial cycle time of fetching the twiddle factors (omegas)
        self.cycle_time = mem_latency

    def step(self, data, tag, tags_only=False):
        """
        Advance the simulator by one time step.
        Both `data` and `tag` are required and go into READ stage.

        Parameters:
        - data: list of integers to feed into READ stage
        - tag: identifier associated with this data
        """
        if data is not None:
            if not isinstance(data, list):
                raise ValueError("Data must be a list of integers or None.")

        # Move COMPUTE result to WRITE
        self.pipeline["WRITE"] = self.pipeline["COMPUTE"]
        self.out = self.pipeline["WRITE"]

        # If there's something in READ, apply NTT and put in COMPUTE
        if self.pipeline["READ"] != (None, None):
            read_data, read_tag = self.pipeline["READ"]
            
            # for functional simulation, we are using fast NTT approach and performing bit-reversal,
            # but not counting latency of bit-reversal because we assume the chaining property will be
            # used external to the NTT functionality
            compute_result = bit_rev_shuffle(ntt_dif_nr(read_data, self.modulus, self.omegas))
            
            self.pipeline["COMPUTE"] = (compute_result, read_tag)
        else:
            self.pipeline["COMPUTE"] = (None, None)

        # Put new data and tag into READ stage
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
        self.cycle_time += max(stage_times)
        print(f"Cycle time: {self.cycle_time}")
        
        # Print the current state with tags only
        print(self.__str__(tags_only))

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
                tag_str = f"col{tag}"
                data_str = str(data)

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